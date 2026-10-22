#ଅନ୍ତର୍ଭାଵ "rope.hpp"
#ଅନ୍ତର୍ଭାଵ "convert.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

ସଂରଚନା rope_corr_dims {
    ଦଶମିକ v[2];
};

ସଂରଚନା mrope_sections {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା v[4];
};

ସ୍ଥିର ଦଶମିକ rope_yarn_ramp(ଅଚଳ ଦଶମିକ ତଳ, ଅଚଳ ଦଶମିକ ଉପର, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0) {
    ଅଚଳ ଦଶମିକ y = (i0 / 2 - ତଳ) / sycl::max(0.001f, ଉପର - ତଳ);
    ଫେରନ୍ତୁ 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <ତାର୍କିକ forward>
ସ୍ଥିର ଶୂନ୍ୟ rope_yarn(ଅଚଳ ଦଶମିକ theta_extrap, ଅଚଳ ଦଶମିକ freq_scale,
                      ଅଚଳ rope_corr_dims corr_dims, ଅଚଳ int64_t i0,
                      ଅଚଳ ଦଶମିକ ext_factor, ଦଶମିକ mscale, ଦଶମିକ &cos_theta,
                      ଦଶମିକ &sin_theta) {
    ଦଶମିକ theta_interp = freq_scale * theta_extrap;
    ଦଶମିକ theta = theta_interp;
    ଯଦି (ext_factor != 0.0f) {
        ଦଶମିକ ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    ଯଦି (!forward) {
        sin_theta *= -1.0f;
    }
}

template <ତାର୍କିକ forward, ତାର୍କିକ has_ff, typename T, typename D>
ସ୍ଥିର ଶୂନ୍ୟ rope_norm(ଅଚଳ T *x, D *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs, ଅଚଳ int32_t *pos,
                      ଅଚଳ ଦଶମିକ freq_scale, ଅଚଳ ଦଶମିକ ext_factor,
                      ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
                      ଅଚଳ ଦଶମିକ theta_scale, ଅଚଳ ଦଶମିକ *freq_factors,
                      ଅଚଳ int64_t *row_indices, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ଯଦି (i0 >= ne00) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ଅଚଳ uint32_t i3 = row_dst / (ne01 * ne02);
    ଅଚଳ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ଅଚଳ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    ଯଦି (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    ଅଚଳ auto &store_coaelsced = [&](ଦଶମିକ x0, ଦଶମିକ x1) {
        ଯଦି constexpr (std::is_same_v<ଦଶମିକ, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    ଯଦି (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = i0 - n_offs; // relative idx

    ଅଚଳ ଦଶମିକ theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ଅଚଳ ଦଶମିକ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ଦଶମିକ cos_theta;
    ଦଶମିକ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ଅଚଳ ଦଶମିକ x0 = x[ix + 0];
    ଅଚଳ ଦଶମିକ x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <ତାର୍କିକ forward, ତାର୍କିକ has_ff, typename T, typename D>
ସ୍ଥିର ଶୂନ୍ୟ rope_neox(ଅଚଳ T *x, D *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs, ଅଚଳ int32_t *pos,
                      ଅଚଳ ଦଶମିକ freq_scale, ଅଚଳ ଦଶମିକ ext_factor,
                      ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
                      ଅଚଳ ଦଶମିକ theta_scale, ଅଚଳ ଦଶମିକ *freq_factors,
                      ଅଚଳ int64_t *row_indices, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ଯଦି (i0 >= ne00) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ଅଚଳ uint32_t i3 = row_dst / (ne01 * ne02);
    ଅଚଳ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ଅଚଳ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ଯଦି (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    ଯଦି (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = i0 - n_offs; // relative idx

    ଅଚଳ ଦଶମିକ theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ଅଚଳ ଦଶମିକ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ଦଶମିକ cos_theta;
    ଦଶମିକ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ଅଚଳ ଦଶମିକ x0 = x[ix + n_offs / 2 + 0];
    ଅଚଳ ଦଶମିକ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <ତାର୍କିକ forward, ତାର୍କିକ has_ff, typename T>
ସ୍ଥିର ଶୂନ୍ୟ rope_multi(ଅଚଳ T *x, T *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02,
                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3,
                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs, ଅଚଳ int32_t *pos,
                       ଅଚଳ ଦଶମିକ freq_scale, ଅଚଳ ଦଶମିକ ext_factor,
                       ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
                       ଅଚଳ ଦଶମିକ theta_scale, ଅଚଳ ଦଶମିକ *freq_factors,
                       ଅଚଳ mrope_sections sections, ଅଚଳ ତାର୍କିକ is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ଯଦି (i0 >= ne00) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ଅଚଳ uint32_t i3 = row_dst / (ne01 * ne02);
    ଅଚଳ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ଅଚଳ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ଯଦି (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = i0 - n_offs; // relative idx

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w = sections.v[1] + sections.v[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sector = (iw / 2) % sect_dims;

    ଦଶମିକ theta_base = 0.0;
    ଯଦି (is_imrope) {
        ଯଦି (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } ଅନ୍ୟଥା ଯଦି (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } ଅନ୍ୟଥା ଯଦି (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } ଅନ୍ୟଥା {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } ଅନ୍ୟଥା ଯଦି (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } ଅନ୍ୟଥା ଯଦି (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } ଅନ୍ୟଥା ଯଦି (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    ଅଚଳ ଦଶମିକ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ଦଶମିକ cos_theta;
    ଦଶମିକ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ଅଚଳ ଦଶମିକ x0 = x[ix + n_offs / 2 + 0];
    ଅଚଳ ଦଶମିକ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <ତାର୍କିକ forward, ତାର୍କିକ has_ff, typename T>
ସ୍ଥିର ଶୂନ୍ୟ rope_vision(ଅଚଳ T *x, T *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02,
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3,
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ଅଚଳ int32_t *pos,
                        ଅଚଳ ଦଶମିକ freq_scale, ଅଚଳ ଦଶମିକ ext_factor,
                        ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
                        ଅଚଳ ଦଶମିକ theta_scale, ଅଚଳ ଦଶମିକ *freq_factors,
                        ଅଚଳ mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ଯଦି (i0 >= ne00) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ଅଚଳ uint32_t i3 = row_dst / (ne01 * ne02);
    ଅଚଳ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ଅଚଳ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sect_dims = sections.v[0] + sections.v[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w = sections.v[1] + sections.v[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sector = (i0 / 2) % sect_dims;

    ଦଶମିକ theta_base = 0.0;
    ଯଦି (sector < sections.v[0]) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } ଅନ୍ୟଥା ଯଦି (sector >= sections.v[0] && sector < sec_w) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    ଅଚଳ ଦଶମିକ freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    ଦଶମିକ cos_theta;
    ଦଶମିକ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ଅଚଳ ଦଶମିକ x0 = x[ix + 0];
    ଅଚଳ ଦଶମିକ x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <ତାର୍କିକ forward, typename T, typename D>
ସ୍ଥିର ଶୂନ୍ୟ
rope_norm_sycl(ଅଚଳ T *x, D *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr, ଅଚଳ int32_t *pos, ଅଚଳ ଦଶମିକ freq_scale,
               ଅଚଳ ଦଶମିକ freq_base, ଅଚଳ ଦଶମିକ ext_factor,
               ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
               ଅଚଳ ଦଶମିକ *freq_factors, ଅଚଳ int64_t *row_indices,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ଅଚଳ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ଅଚଳ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ଅଚଳ ଦଶମିକ theta_scale = powf(freq_base, -2.0f / n_dims);

    ଯଦି (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, ମିଥ୍ୟା>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } ଅନ୍ୟଥା {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, ସତ୍ୟ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <ତାର୍କିକ forward, typename T, typename D>
ସ୍ଥିର ଶୂନ୍ୟ
rope_neox_sycl(ଅଚଳ T *x, D *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr, ଅଚଳ int32_t *pos, ଅଚଳ ଦଶମିକ freq_scale,
               ଅଚଳ ଦଶମିକ freq_base, ଅଚଳ ଦଶମିକ ext_factor,
               ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
               ଅଚଳ ଦଶମିକ *freq_factors, ଅଚଳ int64_t *row_indices,
               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ଅଚଳ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ଅଚଳ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ଅଚଳ ଦଶମିକ theta_scale = powf(freq_base, -2.0f / n_dims);

    ଯଦି (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, ମିଥ୍ୟା>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } ଅନ୍ୟଥା {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, ସତ୍ୟ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <ତାର୍କିକ forward, typename T>
ସ୍ଥିର ଶୂନ୍ୟ
rope_multi_sycl(ଅଚଳ T *x, T *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03,
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims,
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr, ଅଚଳ int32_t *pos, ଅଚଳ ଦଶମିକ freq_scale,
                ଅଚଳ ଦଶମିକ freq_base, ଅଚଳ ଦଶମିକ ext_factor,
                ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
                ଅଚଳ ଦଶମିକ *freq_factors, ଅଚଳ mrope_sections sections,
                ଅଚଳ ତାର୍କିକ is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ଅଚଳ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ଅଚଳ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ଅଚଳ ଦଶମିକ theta_scale = powf(freq_base, -2.0f / n_dims);

    ଯଦି (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, ମିଥ୍ୟା, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } ଅନ୍ୟଥା {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, ସତ୍ୟ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <ତାର୍କିକ forward, typename T>
ସ୍ଥିର ଶୂନ୍ୟ
rope_vision_sycl(ଅଚଳ T *x, T *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                 ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s03,
                 ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims,
                 ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr, ଅଚଳ int32_t *pos, ଅଚଳ ଦଶମିକ freq_scale,
                 ଅଚଳ ଦଶମିକ freq_base, ଅଚଳ ଦଶମିକ ext_factor,
                 ଅଚଳ ଦଶମିକ attn_factor, ଅଚଳ rope_corr_dims corr_dims,
                 ଅଚଳ ଦଶମିକ *freq_factors, ଅଚଳ mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ଅଚଳ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ଅଚଳ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ଅଚଳ ଦଶମିକ theta_scale = powf(freq_base, -2.0f / n_dims);

    ଯଦି (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, ମିଥ୍ୟା, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } ଅନ୍ୟଥା {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, ସତ୍ୟ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <ତାର୍କିକ forward>
ଶୂନ୍ୟ ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ଅଚଳ ggml_tensor *set_rows = nullptr) {
    ଅଚଳ ggml_tensor *src0 = dst->src[0];
    ଅଚଳ ggml_tensor *src1 = dst->src[1];
    ଅଚଳ ggml_tensor *src2 = dst->src[2];

    ଅଚଳ ଦଶମିକ *src0_d = (ଅଚଳ ଦଶମିକ *)src0->data;
    ଅଚଳ ଦଶମିକ *src1_d = (ଅଚଳ ଦଶମିକ *)src1->data;

    ଶୂନ୍ୟ *dst_d = dst->data;
    ଅଚଳ int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_rows_stride = 0;

    ଯଦି (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (ଅଚଳ int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ଅଚଳ int64_t ne00 = src0->ne[0]; // head dims
    ଅଚଳ int64_t ne01 = src0->ne[1]; // num heads
    ଅଚଳ int64_t ne02 = src0->ne[2]; // num heads
    ଅଚଳ int64_t nr = ggml_nrows(src0);

    ଅଚଳ size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    ଅଚଳ size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    ଅଚଳ size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    ଅଚଳ size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    ଅଚଳ size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    ଅଚଳ size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims = ((int32_t *)dst->op_params)[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode = ((int32_t *)dst->op_params)[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    ଦଶମିକ freq_base;
    ଦଶମିକ freq_scale;
    ଦଶମିକ ext_factor;
    ଦଶମିକ attn_factor;
    ଦଶମିକ beta_fast;
    ଦଶମିକ beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, ଆକାର(ଦଶମିକ));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, ଆକାର(ଦଶମିକ));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, ଆକାର(ଦଶମିକ));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, ଆକାର(ଦଶମିକ));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, ଆକାର(ଦଶମିକ));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, ଆକାର(ଦଶମିକ));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) * 4);

    ଅଚଳ ତାର୍କିକ is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ଅଚଳ ତାର୍କିକ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ଅଚଳ ତାର୍କିକ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ଅଚଳ ତାର୍କିକ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ଯଦି (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    ଯଦି (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ଅଚଳ int32_t *pos = (ଅଚଳ int32_t *)src1_d;

    ଅଚଳ ଦଶମିକ *freq_factors = nullptr;
    ଯଦି (src2 != nullptr) {
        freq_factors = (ଅଚଳ ଦଶମିକ *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    ଯଦି (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        ଯଦି (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, ଦଶମିକ, ଦଶମିକ>(
                (ଅଚଳ ଦଶମିକ *)src0_d, (ଦଶମିକ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, ଦଶମିକ, sycl::half>(
                (ଅଚଳ ଦଶମିକ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ଅଚଳ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ଅନ୍ୟଥା {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ଅନ୍ୟଥା ଯଦି (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        ଯଦି (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ଅଚଳ ଦଶମିକ *)src0_d, (ଦଶମିକ *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ଅଚଳ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } ଅନ୍ୟଥା {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ଅନ୍ୟଥା ଯଦି (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        ଯଦି (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ଅଚଳ ଦଶମିକ *)src0_d, (ଦଶମିକ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ଅଚଳ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } ଅନ୍ୟଥା {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ଅନ୍ୟଥା {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        ଯଦି (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, ଦଶମିକ, ଦଶମିକ>(
                (ଅଚଳ ଦଶମିକ *)src0_d, (ଦଶମିକ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, ଦଶମିକ, sycl::half>(
                (ଅଚଳ ଦଶମିକ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ଅଚଳ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ଅନ୍ୟଥା {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

ଶୂନ୍ୟ ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<ସତ୍ୟ>(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ମିଥ୍ୟା>(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ସତ୍ୟ>(ctx, rope, set_rows);
}
