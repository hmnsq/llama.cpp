#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ <float.h>

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*cpy_kernel_t)(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst);

__dpct_inline__ ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_index_int8(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ int8_t * val, ଦଶମିକ x) {
    ଯଦି (x <= val[0]) {
        ଫେରନ୍ତୁ 0;
    }
    ଯଦି (x >= val[n - 1]) {
        ଫେରନ୍ତୁ n - 1;
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ml = 0, mu = n - 1;
    ଯେତେବେଳେ (mu - ml > 1) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା mav = (ml + mu) / 2;
        ଯଦି (x < val[mav]) {
            mu = mav;
        } ଅନ୍ୟଥା {
            ml = mav;
        }
    }
    ଫେରନ୍ତୁ x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q8_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    ଦଶମିକ amax = 0.0f;  // absolute max

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; j++) {
        ଅଚଳ ଦଶମିକ v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((ଦଶମିକ) v));
    }

    ଅଚଳ ଦଶମିକ d  = amax / ((1 << 7) - 1);
    ଅଚଳ ଦଶମିକ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; ++j) {
        ଅଚଳ ଦଶମିକ x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((ଦଶମିକ) x0);
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q1_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    ଦଶମିକ sum_abs = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((ଦଶମିକ) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK1_0; ++j) {
        ଯଦି (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା round_nearest_int(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q2_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    ଦଶମିକ amax = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ଦଶମିକ) xi[j]));
    }

    ଅଚଳ ଦଶମିକ d  = amax;
    ଅଚଳ ଦଶମିକ id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0; ++j) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte_index = j / 4;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_index_mxfp4(ଅଚଳ ଦଶମିକ x, ଅଚଳ ଦଶମିକ e) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_index = 0;
    ଦଶମିକ best_err = sycl::fabs((ଦଶମିକ) (kvalues_mxfp4[0] * e - x));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < 16; ++i) {
        ଅଚଳ ଦଶମିକ err = sycl::fabs((ଦଶମିକ) (kvalues_mxfp4[i] * e - x));
        ଯଦି (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    ଫେରନ୍ତୁ best_index;
}

ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nearest_int_sycl(ଦଶମିକ x) {
    ଅଚଳ ଦଶମିକ val = x + 12582912.0f;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i;
    memcpy(&i, &val, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ଫେରନ୍ତୁ (i & 0x007fffff) - 0x00400000;
}

ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nearest_int_ggml_sycl(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sycl::round((ଦଶମିକ) x);
}

ଧାଡ଼ିଭିତରେ uint8_t clamp_u8(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lo, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା hi) {
    ଫେରନ୍ତୁ (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

ଧାଡ଼ିଭିତରେ int8_t clamp_i8(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lo, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା hi) {
    ଫେରନ୍ତୁ (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr ଦଶମିକ GROUP_MAX_EPS_SYCL = 1e-15f;

ଧାଡ଼ିଭିତରେ ଦଶମିକ make_qx_quants_sycl(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nmax, ଅଚଳ ଦଶମିକ * x, int8_t * L, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rmse_type, ଅଚଳ ଦଶମିକ * qw) {
    ଦଶମିକ max = 0.0f;
    ଦଶମିକ amax = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଅଚଳ ଦଶମିକ ax = sycl::fabs(x[i]);
        ଯଦି (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    ଯଦି (amax < GROUP_MAX_EPS_SYCL) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ଫେରନ୍ତୁ 0.0f;
    }

    ଦଶମିକ iscale = -nmax / max;
    ଯଦି (rmse_type == 0) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ଫେରନ୍ତୁ 1.0f / iscale;
    }

    ତାର୍କିକ return_early = ମିଥ୍ୟା;
    ଯଦି (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = ସତ୍ୟ;
    }

    ଦଶମିକ sumlx = 0.0f;
    ଦଶମିକ suml2 = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ଅଚଳ ଦଶମିକ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    ଦଶମିକ scale = suml2 ? sumlx / suml2 : 0.0f;
    ଯଦି (return_early) {
        ଫେରନ୍ତୁ suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    ଦଶମିକ best = scale * sumlx;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା is = -9; is <= 9; ++is) {
        ଯଦି (is == 0) {
            ଜାରି;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ଅଚଳ ଦଶମିକ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ଯଦି (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    ଫେରନ୍ତୁ scale;
}

ଧାଡ଼ିଭିତରେ ଦଶମିକ make_q3_quants_sycl(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nmax, ଅଚଳ ଦଶମିକ * x, int8_t * L, ତାର୍କିକ do_rmse) {
    ଦଶମିକ max = 0.0f;
    ଦଶମିକ amax = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଅଚଳ ଦଶମିକ ax = sycl::fabs(x[i]);
        ଯଦି (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    ଯଦି (amax < GROUP_MAX_EPS_SYCL) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ଫେରନ୍ତୁ 0.0f;
    }

    ଅଚଳ ଦଶମିକ iscale = -nmax / max;
    ଯଦି (do_rmse) {
        ଦଶମିକ sumlx = 0.0f;
        ଦଶମିକ suml2 = 0.0f;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ଅଚଳ ଦଶମିକ w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା itry = 0; itry < 5; ++itry) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_changed = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
                ଅଚଳ ଦଶମିକ w = x[i] * x[i];
                ଦଶମିକ slx = sumlx - w * x[i] * L[i];
                ଯଦି (slx > 0.0f) {
                    ଦଶମିକ sl2 = suml2 - w * L[i] * L[i];
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    ଯଦି (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        ଯଦି (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            ଯଦି (!n_changed) {
                ଭାଙ୍ଗ;
            }
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ଫେରନ୍ତୁ suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ଫେରନ୍ତୁ 1.0f / iscale;
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ set_scale_min_k4(ପୂର୍ଣ୍ଣସଂଖ୍ୟା j, uint8_t * q, uint8_t d, uint8_t m) {
    ଯଦି (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } ଅନ୍ୟଥା {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ get_scale_min_k4_local(ପୂର୍ଣ୍ଣସଂଖ୍ୟା j, ଅଚଳ uint8_t * q, uint8_t & d, uint8_t & m) {
    ଯଦି (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ଅନ୍ୟଥା {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_mxfp4(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ *   xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    ଦଶମିକ amax = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ଦଶମିକ) xi[j]));
    }

    ଅଚଳ uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ଅଚଳ ଦଶମିକ d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_MXFP4 / 2; ++j) {
        ଅଚଳ uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        ଅଚଳ uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_nvfp4(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ *   xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < n_sub; ++s) {
        ଅଚଳ ଦଶମିକ * xb = xi + s * QK_NVFP4_SUB;

        ଦଶମିକ amax = 0.0f;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((ଦଶମିକ) xb[j]));
        }

        ଅଚଳ uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ଅଚଳ ଦଶମିକ d = ggml_sycl_ue4m3_to_fp32(ue);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ଅଚଳ uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            ଅଚଳ uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q4_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    ଦଶମିକ amax = 0.0f;
    ଦଶମିକ vmax = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_0; ++j) {
        ଅଚଳ ଦଶମିକ v = xi[j];
        ଯଦି (amax < sycl::fabs((ଦଶମିକ) v)) {
            amax = sycl::fabs((ଦଶମିକ) v);
            vmax = v;
        }
    }

    ଅଚଳ ଦଶମିକ d  = vmax / -8;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_0 / 2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = xi[0 + j] * id;
        ଅଚଳ ଦଶମିକ x1 = xi[QK4_0 / 2 + j] * id;

        ଅଚଳ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ଅଚଳ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q4_1(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    ଦଶମିକ vmin = FLT_MAX;
    ଦଶମିକ vmax = -FLT_MAX;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_1; ++j) {
        ଅଚଳ ଦଶମିକ v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ଅଚଳ ଦଶମିକ d  = (vmax - vmin) / ((1 << 4) - 1);
    ଅଚଳ ଦଶମିକ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_1 / 2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = (xi[0 + j] - vmin) * id;
        ଅଚଳ ଦଶମିକ x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ଅଚଳ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ଅଚଳ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q5_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    ଦଶମିକ amax = 0.0f;
    ଦଶମିକ vmax = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_0; ++j) {
        ଅଚଳ ଦଶମିକ v = xi[j];
        ଯଦି (amax < sycl::fabs((ଦଶମିକ) v)) {
            amax = sycl::fabs((ଦଶମିକ) v);
            vmax = v;
        }
    }

    ଅଚଳ ଦଶମିକ d  = vmax / -16;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_0 / 2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = xi[0 + j] * id;
        ଅଚଳ ଦଶମିକ x1 = xi[QK5_0 / 2 + j] * id;

        ଅଚଳ uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ଅଚଳ uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, ଆକାର(qh));
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_q5_1(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    ଦଶମିକ min = xi[0];
    ଦଶମିକ max = xi[0];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < QK5_1; ++j) {
        ଅଚଳ ଦଶମିକ v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ଅଚଳ ଦଶମିକ d  = (max - min) / 31;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_1 / 2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = (xi[0 + j] - min) * id;
        ଅଚଳ ଦଶମିକ x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ଅଚଳ uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ଅଚଳ uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, ଆକାର(qh));
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ cpy_blck_f32_iq4_nl(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ *  xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    ଦଶମିକ amax = 0.0f;
    ଦଶମିକ vmax = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_NL; ++j) {
        ଅଚଳ ଦଶମିକ v = xi[j];
        ଯଦି (amax < sycl::fabs((ଦଶମିକ) v)) {
            amax = sycl::fabs((ଦଶମିକ) v);
            vmax = v;
        }
    }

    ଦଶମିକ       d  = vmax / kvalues_iq4nl[0];
    ଅଚଳ ଦଶମିକ id = d ? 1.0f / d : 0.0f;

    ଦଶମିକ sumqx = 0, sumq2 = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_NL / 2; ++j) {
        ଅଚଳ ଦଶମିକ   x0  = xi[0 + j] * id;
        ଅଚଳ ଦଶମିକ   x1  = xi[QK4_NL / 2 + j] * id;
        ଅଚଳ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        ଅଚଳ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ଅଚଳ ଦଶମିକ v0    = kvalues_iq4nl[xi0];
        ଅଚଳ ଦଶମିକ v1    = kvalues_iq4nl[xi1];
        ଅଚଳ ଦଶମିକ w0    = xi[0 + j] * xi[0 + j];
        ଅଚଳ ଦଶମିକ w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

ଶୂନ୍ୟ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1);
ଶୂନ୍ୟ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
