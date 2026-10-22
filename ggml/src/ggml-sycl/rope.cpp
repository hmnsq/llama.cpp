#അന്തര്ഭാവ "rope.hpp"
#അന്തര്ഭാവ "convert.hpp"
#അന്തര്ഭാവ "ggml-sycl/common.hpp"
#അന്തര്ഭാവ "ggml.h"

ഘടന rope_corr_dims {
    ദശാംശം v[2];
};

ഘടന mrope_sections {
    പൂർണ്ണസംഖ്യ v[4];
};

സ്ഥിതം ദശാംശം rope_yarn_ramp(സ്ഥിരം ദശാംശം താഴെയുള്ള, സ്ഥിരം ദശാംശം മുകളിലെ, സ്ഥിരം പൂർണ്ണസംഖ്യ i0) {
    സ്ഥിരം ദശാംശം y = (i0 / 2 - താഴെയുള്ള) / sycl::max(0.001f, മുകളിലെ - താഴെയുള്ള);
    തിരികെ 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <തർക്കം forward>
സ്ഥിതം ശൂന്യം rope_yarn(സ്ഥിരം ദശാംശം theta_extrap, സ്ഥിരം ദശാംശം freq_scale,
                      സ്ഥിരം rope_corr_dims corr_dims, സ്ഥിരം int64_t i0,
                      സ്ഥിരം ദശാംശം ext_factor, ദശാംശം mscale, ദശാംശം &cos_theta,
                      ദശാംശം &sin_theta) {
    ദശാംശം theta_interp = freq_scale * theta_extrap;
    ദശാംശം theta = theta_interp;
    എങ്കിൽ (ext_factor != 0.0f) {
        ദശാംശം ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    എങ്കിൽ (!forward) {
        sin_theta *= -1.0f;
    }
}

template <തർക്കം forward, തർക്കം has_ff, typename T, typename D>
സ്ഥിതം ശൂന്യം rope_norm(സ്ഥിരം T *x, D *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02,
                      സ്ഥിരം പൂർണ്ണസംഖ്യ s03, സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3,
                      സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims, സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs, സ്ഥിരം int32_t *pos,
                      സ്ഥിരം ദശാംശം freq_scale, സ്ഥിരം ദശാംശം ext_factor,
                      സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
                      സ്ഥിരം ദശാംശം theta_scale, സ്ഥിരം ദശാംശം *freq_factors,
                      സ്ഥിരം int64_t *row_indices, സ്ഥിരം പൂർണ്ണസംഖ്യ set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    എങ്കിൽ (i0 >= ne00) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    സ്ഥിരം uint32_t i3 = row_dst / (ne01 * ne02);
    സ്ഥിരം uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    സ്ഥിരം uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    പൂർണ്ണസംഖ്യ idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    എങ്കിൽ (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    സ്ഥിരം auto &store_coaelsced = [&](ദശാംശം x0, ദശാംശം x1) {
        എങ്കിൽ constexpr (std::is_same_v<ദശാംശം, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    എങ്കിൽ (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ iw = i0 - n_offs; // relative idx

    സ്ഥിരം ദശാംശം theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    സ്ഥിരം ദശാംശം freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ദശാംശം cos_theta;
    ദശാംശം sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    സ്ഥിരം ദശാംശം x0 = x[ix + 0];
    സ്ഥിരം ദശാംശം x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <തർക്കം forward, തർക്കം has_ff, typename T, typename D>
സ്ഥിതം ശൂന്യം rope_neox(സ്ഥിരം T *x, D *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02,
                      സ്ഥിരം പൂർണ്ണസംഖ്യ s03, സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3,
                      സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims, സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs, സ്ഥിരം int32_t *pos,
                      സ്ഥിരം ദശാംശം freq_scale, സ്ഥിരം ദശാംശം ext_factor,
                      സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
                      സ്ഥിരം ദശാംശം theta_scale, സ്ഥിരം ദശാംശം *freq_factors,
                      സ്ഥിരം int64_t *row_indices, സ്ഥിരം പൂർണ്ണസംഖ്യ set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    എങ്കിൽ (i0 >= ne00) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    സ്ഥിരം uint32_t i3 = row_dst / (ne01 * ne02);
    സ്ഥിരം uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    സ്ഥിരം uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    പൂർണ്ണസംഖ്യ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    എങ്കിൽ (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    എങ്കിൽ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ iw = i0 - n_offs; // relative idx

    സ്ഥിരം ദശാംശം theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    സ്ഥിരം ദശാംശം freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ദശാംശം cos_theta;
    ദശാംശം sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    സ്ഥിരം ദശാംശം x0 = x[ix + n_offs / 2 + 0];
    സ്ഥിരം ദശാംശം x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <തർക്കം forward, തർക്കം has_ff, typename T>
സ്ഥിതം ശൂന്യം rope_multi(സ്ഥിരം T *x, T *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                       സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02,
                       സ്ഥിരം പൂർണ്ണസംഖ്യ s03, സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3,
                       സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims, സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs, സ്ഥിരം int32_t *pos,
                       സ്ഥിരം ദശാംശം freq_scale, സ്ഥിരം ദശാംശം ext_factor,
                       സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
                       സ്ഥിരം ദശാംശം theta_scale, സ്ഥിരം ദശാംശം *freq_factors,
                       സ്ഥിരം mrope_sections sections, സ്ഥിരം തർക്കം is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    എങ്കിൽ (i0 >= ne00) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    സ്ഥിരം uint32_t i3 = row_dst / (ne01 * ne02);
    സ്ഥിരം uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    സ്ഥിരം uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    പൂർണ്ണസംഖ്യ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    എങ്കിൽ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ iw = i0 - n_offs; // relative idx

    സ്ഥിരം പൂർണ്ണസംഖ്യ sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ sec_w = sections.v[1] + sections.v[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ sector = (iw / 2) % sect_dims;

    ദശാംശം theta_base = 0.0;
    എങ്കിൽ (is_imrope) {
        എങ്കിൽ (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } അല്ലെങ്കിൽ {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    സ്ഥിരം ദശാംശം freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ദശാംശം cos_theta;
    ദശാംശം sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    സ്ഥിരം ദശാംശം x0 = x[ix + n_offs / 2 + 0];
    സ്ഥിരം ദശാംശം x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <തർക്കം forward, തർക്കം has_ff, typename T>
സ്ഥിതം ശൂന്യം rope_vision(സ്ഥിരം T *x, T *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                        സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02,
                        സ്ഥിരം പൂർണ്ണസംഖ്യ s03, സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3,
                        സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims, സ്ഥിരം int32_t *pos,
                        സ്ഥിരം ദശാംശം freq_scale, സ്ഥിരം ദശാംശം ext_factor,
                        സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
                        സ്ഥിരം ദശാംശം theta_scale, സ്ഥിരം ദശാംശം *freq_factors,
                        സ്ഥിരം mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    എങ്കിൽ (i0 >= ne00) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    സ്ഥിരം uint32_t i3 = row_dst / (ne01 * ne02);
    സ്ഥിരം uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    സ്ഥിരം uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    പൂർണ്ണസംഖ്യ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    സ്ഥിരം പൂർണ്ണസംഖ്യ sect_dims = sections.v[0] + sections.v[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ sec_w = sections.v[1] + sections.v[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ sector = (i0 / 2) % sect_dims;

    ദശാംശം theta_base = 0.0;
    എങ്കിൽ (sector < sections.v[0]) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } അല്ലെങ്കിൽ എങ്കിൽ (sector >= sections.v[0] && sector < sec_w) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    സ്ഥിരം ദശാംശം freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    ദശാംശം cos_theta;
    ദശാംശം sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    സ്ഥിരം ദശാംശം x0 = x[ix + 0];
    സ്ഥിരം ദശാംശം x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <തർക്കം forward, typename T, typename D>
സ്ഥിതം ശൂന്യം
rope_norm_sycl(സ്ഥിരം T *x, D *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
               സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02, സ്ഥിരം പൂർണ്ണസംഖ്യ s03,
               സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3, സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims,
               സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs, സ്ഥിരം പൂർണ്ണസംഖ്യ nr, സ്ഥിരം int32_t *pos, സ്ഥിരം ദശാംശം freq_scale,
               സ്ഥിരം ദശാംശം freq_base, സ്ഥിരം ദശാംശം ext_factor,
               സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
               സ്ഥിരം ദശാംശം *freq_factors, സ്ഥിരം int64_t *row_indices,
               സ്ഥിരം പൂർണ്ണസംഖ്യ set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    സ്ഥിരം dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    സ്ഥിരം dpct::dim3 block_nums(nr, n_blocks_x, 1);

    സ്ഥിരം ദശാംശം theta_scale = powf(freq_base, -2.0f / n_dims);

    എങ്കിൽ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, അസത്യം>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } അല്ലെങ്കിൽ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, സത്യം>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <തർക്കം forward, typename T, typename D>
സ്ഥിതം ശൂന്യം
rope_neox_sycl(സ്ഥിരം T *x, D *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
               സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02, സ്ഥിരം പൂർണ്ണസംഖ്യ s03,
               സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3, സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims,
               സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs, സ്ഥിരം പൂർണ്ണസംഖ്യ nr, സ്ഥിരം int32_t *pos, സ്ഥിരം ദശാംശം freq_scale,
               സ്ഥിരം ദശാംശം freq_base, സ്ഥിരം ദശാംശം ext_factor,
               സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
               സ്ഥിരം ദശാംശം *freq_factors, സ്ഥിരം int64_t *row_indices,
               സ്ഥിരം പൂർണ്ണസംഖ്യ set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    സ്ഥിരം dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    സ്ഥിരം dpct::dim3 block_nums(nr, n_blocks_x, 1);

    സ്ഥിരം ദശാംശം theta_scale = powf(freq_base, -2.0f / n_dims);

    എങ്കിൽ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, അസത്യം>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } അല്ലെങ്കിൽ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, സത്യം>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <തർക്കം forward, typename T>
സ്ഥിതം ശൂന്യം
rope_multi_sycl(സ്ഥിരം T *x, T *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02, സ്ഥിരം പൂർണ്ണസംഖ്യ s03,
                സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3, സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims,
                സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs, സ്ഥിരം പൂർണ്ണസംഖ്യ nr, സ്ഥിരം int32_t *pos, സ്ഥിരം ദശാംശം freq_scale,
                സ്ഥിരം ദശാംശം freq_base, സ്ഥിരം ദശാംശം ext_factor,
                സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
                സ്ഥിരം ദശാംശം *freq_factors, സ്ഥിരം mrope_sections sections,
                സ്ഥിരം തർക്കം is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    സ്ഥിരം dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    സ്ഥിരം dpct::dim3 block_nums(nr, n_blocks_x, 1);

    സ്ഥിരം ദശാംശം theta_scale = powf(freq_base, -2.0f / n_dims);

    എങ്കിൽ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, അസത്യം, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } അല്ലെങ്കിൽ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, സത്യം, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <തർക്കം forward, typename T>
സ്ഥിതം ശൂന്യം
rope_vision_sycl(സ്ഥിരം T *x, T *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                 സ്ഥിരം പൂർണ്ണസംഖ്യ ne02, സ്ഥിരം പൂർണ്ണസംഖ്യ s01, സ്ഥിരം പൂർണ്ണസംഖ്യ s02, സ്ഥിരം പൂർണ്ണസംഖ്യ s03,
                 സ്ഥിരം പൂർണ്ണസംഖ്യ s1, സ്ഥിരം പൂർണ്ണസംഖ്യ s2, സ്ഥിരം പൂർണ്ണസംഖ്യ s3, സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims,
                 സ്ഥിരം പൂർണ്ണസംഖ്യ nr, സ്ഥിരം int32_t *pos, സ്ഥിരം ദശാംശം freq_scale,
                 സ്ഥിരം ദശാംശം freq_base, സ്ഥിരം ദശാംശം ext_factor,
                 സ്ഥിരം ദശാംശം attn_factor, സ്ഥിരം rope_corr_dims corr_dims,
                 സ്ഥിരം ദശാംശം *freq_factors, സ്ഥിരം mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    സ്ഥിരം dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    സ്ഥിരം dpct::dim3 block_nums(nr, n_blocks_x, 1);

    സ്ഥിരം ദശാംശം theta_scale = powf(freq_base, -2.0f / n_dims);

    എങ്കിൽ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, അസത്യം, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } അല്ലെങ്കിൽ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, സത്യം, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <തർക്കം forward>
ശൂന്യം ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            സ്ഥിരം ggml_tensor *set_rows = nullptr) {
    സ്ഥിരം ggml_tensor *src0 = dst->src[0];
    സ്ഥിരം ggml_tensor *src1 = dst->src[1];
    സ്ഥിരം ggml_tensor *src2 = dst->src[2];

    സ്ഥിരം ദശാംശം *src0_d = (സ്ഥിരം ദശാംശം *)src0->data;
    സ്ഥിരം ദശാംശം *src1_d = (സ്ഥിരം ദശാംശം *)src1->data;

    ശൂന്യം *dst_d = dst->data;
    സ്ഥിരം int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    പൂർണ്ണസംഖ്യ set_rows_stride = 0;

    എങ്കിൽ (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (സ്ഥിരം int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    സ്ഥിരം int64_t ne00 = src0->ne[0]; // head dims
    സ്ഥിരം int64_t ne01 = src0->ne[1]; // num heads
    സ്ഥിരം int64_t ne02 = src0->ne[2]; // num heads
    സ്ഥിരം int64_t nr = ggml_nrows(src0);

    സ്ഥിരം size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    സ്ഥിരം size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    സ്ഥിരം size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    സ്ഥിരം size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    സ്ഥിരം size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    സ്ഥിരം size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims = ((int32_t *)dst->op_params)[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ mode = ((int32_t *)dst->op_params)[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_ctx_orig = ((int32_t *)dst->op_params)[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    ദശാംശം freq_base;
    ദശാംശം freq_scale;
    ദശാംശം ext_factor;
    ദശാംശം attn_factor;
    ദശാംശം beta_fast;
    ദശാംശം beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, വലുപ്പം(ദശാംശം));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, വലുപ്പം(ദശാംശം));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, വലുപ്പം(ദശാംശം));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, വലുപ്പം(ദശാംശം));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, വലുപ്പം(ദശാംശം));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, വലുപ്പം(ദശാംശം));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, വലുപ്പം(പൂർണ്ണസംഖ്യ) * 4);

    സ്ഥിരം തർക്കം is_neox = mode & GGML_ROPE_TYPE_NEOX;
    സ്ഥിരം തർക്കം is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    സ്ഥിരം തർക്കം is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    സ്ഥിരം തർക്കം is_vision = mode == GGML_ROPE_TYPE_VISION;

    എങ്കിൽ (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    എങ്കിൽ (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    സ്ഥിരം int32_t *pos = (സ്ഥിരം int32_t *)src1_d;

    സ്ഥിരം ദശാംശം *freq_factors = nullptr;
    എങ്കിൽ (src2 != nullptr) {
        freq_factors = (സ്ഥിരം ദശാംശം *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    എങ്കിൽ (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        എങ്കിൽ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, ദശാംശം, ദശാംശം>(
                (സ്ഥിരം ദശാംശം *)src0_d, (ദശാംശം *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, ദശാംശം, sycl::half>(
                (സ്ഥിരം ദശാംശം *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (സ്ഥിരം sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } അല്ലെങ്കിൽ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        എങ്കിൽ (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((സ്ഥിരം ദശാംശം *)src0_d, (ദശാംശം *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (സ്ഥിരം sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } അല്ലെങ്കിൽ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        എങ്കിൽ (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (സ്ഥിരം ദശാംശം *)src0_d, (ദശാംശം *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (സ്ഥിരം sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } അല്ലെങ്കിൽ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } അല്ലെങ്കിൽ {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        എങ്കിൽ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, ദശാംശം, ദശാംശം>(
                (സ്ഥിരം ദശാംശം *)src0_d, (ദശാംശം *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, ദശാംശം, sycl::half>(
                (സ്ഥിരം ദശാംശം *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (സ്ഥിരം sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } അല്ലെങ്കിൽ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

ശൂന്യം ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<സത്യം>(ctx, dst);
}

ശൂന്യം ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<അസത്യം>(ctx, dst);
}

ശൂന്യം ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<സത്യം>(ctx, rope, set_rows);
}
