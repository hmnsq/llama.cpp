#ಅನ್ತರ್ಭಾವ "rope.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"

ಸಂರಚನಾ rope_corr_dims {
    ಭಿನ್ನ v[2];
};

ಸಂರಚನಾ mrope_sections {
    ಪೂರ್ಣಾಙ್ಕ v[4];
};

ಸ್ಥಿರ ಭಿನ್ನ rope_yarn_ramp(ಧ್ರುವ ಭಿನ್ನ ಕೆಳಗಿನ, ಧ್ರುವ ಭಿನ್ನ ಮೇಲಿನ, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0) {
    ಧ್ರುವ ಭಿನ್ನ y = (i0 / 2 - ಕೆಳಗಿನ) / sycl::max(0.001f, ಮೇಲಿನ - ಕೆಳಗಿನ);
    ಪ್ರತ್ಯಾಗಮ 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <ತಾರ್ಕಿಕ forward>
ಸ್ಥಿರ ಶೂನ್ಯ rope_yarn(ಧ್ರುವ ಭಿನ್ನ theta_extrap, ಧ್ರುವ ಭಿನ್ನ freq_scale,
                      ಧ್ರುವ rope_corr_dims corr_dims, ಧ್ರುವ int64_t i0,
                      ಧ್ರುವ ಭಿನ್ನ ext_factor, ಭಿನ್ನ mscale, ಭಿನ್ನ &cos_theta,
                      ಭಿನ್ನ &sin_theta) {
    ಭಿನ್ನ theta_interp = freq_scale * theta_extrap;
    ಭಿನ್ನ theta = theta_interp;
    ಯದಿ (ext_factor != 0.0f) {
        ಭಿನ್ನ ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    ಯದಿ (!forward) {
        sin_theta *= -1.0f;
    }
}

template <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, typename T, typename D>
ಸ್ಥಿರ ಶೂನ್ಯ rope_norm(ಧ್ರುವ T *x, D *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs, ಧ್ರುವ int32_t *pos,
                      ಧ್ರುವ ಭಿನ್ನ freq_scale, ಧ್ರುವ ಭಿನ್ನ ext_factor,
                      ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
                      ಧ್ರುವ ಭಿನ್ನ theta_scale, ಧ್ರುವ ಭಿನ್ನ *freq_factors,
                      ಧ್ರುವ int64_t *row_indices, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ಯದಿ (i0 >= ne00) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ಧ್ರುವ uint32_t i3 = row_dst / (ne01 * ne02);
    ಧ್ರುವ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ಧ್ರುವ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಙ್ಕ idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    ಯದಿ (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    ಧ್ರುವ auto &store_coaelsced = [&](ಭಿನ್ನ x0, ಭಿನ್ನ x1) {
        ಯದಿ constexpr (std::is_same_v<ಭಿನ್ನ, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    ಯದಿ (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iw = i0 - n_offs; // relative idx

    ಧ್ರುವ ಭಿನ್ನ theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ಧ್ರುವ ಭಿನ್ನ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ಭಿನ್ನ cos_theta;
    ಭಿನ್ನ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ಧ್ರುವ ಭಿನ್ನ x0 = x[ix + 0];
    ಧ್ರುವ ಭಿನ್ನ x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, typename T, typename D>
ಸ್ಥಿರ ಶೂನ್ಯ rope_neox(ಧ್ರುವ T *x, D *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs, ಧ್ರುವ int32_t *pos,
                      ಧ್ರುವ ಭಿನ್ನ freq_scale, ಧ್ರುವ ಭಿನ್ನ ext_factor,
                      ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
                      ಧ್ರುವ ಭಿನ್ನ theta_scale, ಧ್ರುವ ಭಿನ್ನ *freq_factors,
                      ಧ್ರುವ int64_t *row_indices, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ಯದಿ (i0 >= ne00) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ಧ್ರುವ uint32_t i3 = row_dst / (ne01 * ne02);
    ಧ್ರುವ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ಧ್ರುವ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಙ್ಕ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ಯದಿ (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    ಯದಿ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iw = i0 - n_offs; // relative idx

    ಧ್ರುವ ಭಿನ್ನ theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ಧ್ರುವ ಭಿನ್ನ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ಭಿನ್ನ cos_theta;
    ಭಿನ್ನ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ಧ್ರುವ ಭಿನ್ನ x0 = x[ix + n_offs / 2 + 0];
    ಧ್ರುವ ಭಿನ್ನ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ rope_multi(ಧ್ರುವ T *x, T *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02,
                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3,
                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs, ಧ್ರುವ int32_t *pos,
                       ಧ್ರುವ ಭಿನ್ನ freq_scale, ಧ್ರುವ ಭಿನ್ನ ext_factor,
                       ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
                       ಧ್ರುವ ಭಿನ್ನ theta_scale, ಧ್ರುವ ಭಿನ್ನ *freq_factors,
                       ಧ್ರುವ mrope_sections sections, ಧ್ರುವ ತಾರ್ಕಿಕ is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ಯದಿ (i0 >= ne00) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ಧ್ರುವ uint32_t i3 = row_dst / (ne01 * ne02);
    ಧ್ರುವ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ಧ್ರುವ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಙ್ಕ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ಯದಿ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iw = i0 - n_offs; // relative idx

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sec_w = sections.v[1] + sections.v[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sector = (iw / 2) % sect_dims;

    ಭಿನ್ನ theta_base = 0.0;
    ಯದಿ (is_imrope) {
        ಯದಿ (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } ಅನ್ಯಥಾ ಯದಿ (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } ಅನ್ಯಥಾ ಯದಿ (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } ಅನ್ಯಥಾ {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } ಅನ್ಯಥಾ ಯದಿ (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } ಅನ್ಯಥಾ ಯದಿ (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } ಅನ್ಯಥಾ ಯದಿ (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    ಧ್ರುವ ಭಿನ್ನ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ಭಿನ್ನ cos_theta;
    ಭಿನ್ನ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ಧ್ರುವ ಭಿನ್ನ x0 = x[ix + n_offs / 2 + 0];
    ಧ್ರುವ ಭಿನ್ನ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ rope_vision(ಧ್ರುವ T *x, T *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02,
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3,
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims, ಧ್ರುವ int32_t *pos,
                        ಧ್ರುವ ಭಿನ್ನ freq_scale, ಧ್ರುವ ಭಿನ್ನ ext_factor,
                        ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
                        ಧ್ರುವ ಭಿನ್ನ theta_scale, ಧ್ರುವ ಭಿನ್ನ *freq_factors,
                        ಧ್ರುವ mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    ಯದಿ (i0 >= ne00) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ಧ್ರುವ uint32_t i3 = row_dst / (ne01 * ne02);
    ಧ್ರುವ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ಧ್ರುವ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಙ್ಕ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sect_dims = sections.v[0] + sections.v[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sec_w = sections.v[1] + sections.v[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sector = (i0 / 2) % sect_dims;

    ಭಿನ್ನ theta_base = 0.0;
    ಯದಿ (sector < sections.v[0]) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } ಅನ್ಯಥಾ ಯದಿ (sector >= sections.v[0] && sector < sec_w) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    ಧ್ರುವ ಭಿನ್ನ freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    ಭಿನ್ನ cos_theta;
    ಭಿನ್ನ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ಧ್ರುವ ಭಿನ್ನ x0 = x[ix + 0];
    ಧ್ರುವ ಭಿನ್ನ x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <ತಾರ್ಕಿಕ forward, typename T, typename D>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_norm_sycl(ಧ್ರುವ T *x, D *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr, ಧ್ರುವ int32_t *pos, ಧ್ರುವ ಭಿನ್ನ freq_scale,
               ಧ್ರುವ ಭಿನ್ನ freq_base, ಧ್ರುವ ಭಿನ್ನ ext_factor,
               ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
               ಧ್ರುವ ಭಿನ್ನ *freq_factors, ಧ್ರುವ int64_t *row_indices,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಧ್ರುವ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ಧ್ರುವ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ಧ್ರುವ ಭಿನ್ನ theta_scale = powf(freq_base, -2.0f / n_dims);

    ಯದಿ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, ಅಸತ್ಯ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } ಅನ್ಯಥಾ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, ಸತ್ಯ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <ತಾರ್ಕಿಕ forward, typename T, typename D>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_neox_sycl(ಧ್ರುವ T *x, D *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr, ಧ್ರುವ int32_t *pos, ಧ್ರುವ ಭಿನ್ನ freq_scale,
               ಧ್ರುವ ಭಿನ್ನ freq_base, ಧ್ರುವ ಭಿನ್ನ ext_factor,
               ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
               ಧ್ರುವ ಭಿನ್ನ *freq_factors, ಧ್ರುವ int64_t *row_indices,
               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಧ್ರುವ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ಧ್ರುವ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ಧ್ರುವ ಭಿನ್ನ theta_scale = powf(freq_base, -2.0f / n_dims);

    ಯದಿ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, ಅಸತ್ಯ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } ಅನ್ಯಥಾ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, ಸತ್ಯ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <ತಾರ್ಕಿಕ forward, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_multi_sycl(ಧ್ರುವ T *x, T *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03,
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims,
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr, ಧ್ರುವ int32_t *pos, ಧ್ರುವ ಭಿನ್ನ freq_scale,
                ಧ್ರುವ ಭಿನ್ನ freq_base, ಧ್ರುವ ಭಿನ್ನ ext_factor,
                ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
                ಧ್ರುವ ಭಿನ್ನ *freq_factors, ಧ್ರುವ mrope_sections sections,
                ಧ್ರುವ ತಾರ್ಕಿಕ is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಧ್ರುವ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ಧ್ರುವ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ಧ್ರುವ ಭಿನ್ನ theta_scale = powf(freq_base, -2.0f / n_dims);

    ಯದಿ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, ಅಸತ್ಯ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } ಅನ್ಯಥಾ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, ಸತ್ಯ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <ತಾರ್ಕಿಕ forward, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_vision_sycl(ಧ್ರುವ T *x, T *dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                 ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s03,
                 ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims,
                 ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr, ಧ್ರುವ int32_t *pos, ಧ್ರುವ ಭಿನ್ನ freq_scale,
                 ಧ್ರುವ ಭಿನ್ನ freq_base, ಧ್ರುವ ಭಿನ್ನ ext_factor,
                 ಧ್ರುವ ಭಿನ್ನ attn_factor, ಧ್ರುವ rope_corr_dims corr_dims,
                 ಧ್ರುವ ಭಿನ್ನ *freq_factors, ಧ್ರುವ mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಧ್ರುವ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ಧ್ರುವ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ಧ್ರುವ ಭಿನ್ನ theta_scale = powf(freq_base, -2.0f / n_dims);

    ಯದಿ (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, ಅಸತ್ಯ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } ಅನ್ಯಥಾ {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, ಸತ್ಯ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <ತಾರ್ಕಿಕ forward>
ಶೂನ್ಯ ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ಧ್ರುವ ggml_tensor *set_rows = nullptr) {
    ಧ್ರುವ ggml_tensor *src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor *src1 = dst->src[1];
    ಧ್ರುವ ggml_tensor *src2 = dst->src[2];

    ಧ್ರುವ ಭಿನ್ನ *src0_d = (ಧ್ರುವ ಭಿನ್ನ *)src0->data;
    ಧ್ರುವ ಭಿನ್ನ *src1_d = (ಧ್ರುವ ಭಿನ್ನ *)src1->data;

    ಶೂನ್ಯ *dst_d = dst->data;
    ಧ್ರುವ int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    ಪೂರ್ಣಾಙ್ಕ set_rows_stride = 0;

    ಯದಿ (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (ಧ್ರುವ int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ಧ್ರುವ int64_t ne00 = src0->ne[0]; // head dims
    ಧ್ರುವ int64_t ne01 = src0->ne[1]; // num heads
    ಧ್ರುವ int64_t ne02 = src0->ne[2]; // num heads
    ಧ್ರುವ int64_t nr = ggml_nrows(src0);

    ಧ್ರುವ size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    ಧ್ರುವ size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    ಧ್ರುವ size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    ಧ್ರುವ size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    ಧ್ರುವ size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    ಧ್ರುವ size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims = ((int32_t *)dst->op_params)[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = ((int32_t *)dst->op_params)[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    ಭಿನ್ನ freq_base;
    ಭಿನ್ನ freq_scale;
    ಭಿನ್ನ ext_factor;
    ಭಿನ್ನ attn_factor;
    ಭಿನ್ನ beta_fast;
    ಭಿನ್ನ beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * 4);

    ಧ್ರುವ ತಾರ್ಕಿಕ is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ಯದಿ (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    ಯದಿ (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ಧ್ರುವ int32_t *pos = (ಧ್ರುವ int32_t *)src1_d;

    ಧ್ರುವ ಭಿನ್ನ *freq_factors = nullptr;
    ಯದಿ (src2 != nullptr) {
        freq_factors = (ಧ್ರುವ ಭಿನ್ನ *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    ಯದಿ (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, ಭಿನ್ನ, ಭಿನ್ನ>(
                (ಧ್ರುವ ಭಿನ್ನ *)src0_d, (ಭಿನ್ನ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, ಭಿನ್ನ, sycl::half>(
                (ಧ್ರುವ ಭಿನ್ನ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ಧ್ರುವ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ಅನ್ಯಥಾ ಯದಿ (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        ಯದಿ (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ಧ್ರುವ ಭಿನ್ನ *)src0_d, (ಭಿನ್ನ *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ಧ್ರುವ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ಅನ್ಯಥಾ ಯದಿ (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        ಯದಿ (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ಧ್ರುವ ಭಿನ್ನ *)src0_d, (ಭಿನ್ನ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ಧ್ರುವ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ಅನ್ಯಥಾ {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, ಭಿನ್ನ, ಭಿನ್ನ>(
                (ಧ್ರುವ ಭಿನ್ನ *)src0_d, (ಭಿನ್ನ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, ಭಿನ್ನ, sycl::half>(
                (ಧ್ರುವ ಭಿನ್ನ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ಧ್ರುವ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

ಶೂನ್ಯ ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<ಸತ್ಯ>(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ಅಸತ್ಯ>(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ಸತ್ಯ>(ctx, rope, set_rows);
}
