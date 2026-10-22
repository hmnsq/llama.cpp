#ಅನ್ತರ್ಭಾವ "rope.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"

ರಚನೆ rope_corr_dims {
    ದಶಮಾಂಶ v[2];
};

ರಚನೆ mrope_sections {
    ಪೂರ್ಣಾಂಕ v[4];
};

ಸ್ಥಿರ ದಶಮಾಂಶ rope_yarn_ramp(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲಿನ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = (i0 / 2 - ಕೆಳಗಿನ) / sycl::max(0.001f, ಮೇಲಿನ - ಕೆಳಗಿನ);
    ಹಿಂತಿರುಗಿ 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward>
ಸ್ಥಿರ ಶೂನ್ಯ rope_yarn(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_extrap, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale,
                      ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims, ಸ್ಥಿರಾಂಕ int64_t i0,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ, ದಶಮಾಂಶ mscale, ದಶಮಾಂಶ &cos_theta,
                      ದಶಮಾಂಶ &sin_theta) {
    ದಶಮಾಂಶ theta_interp = freq_scale * theta_extrap;
    ದಶಮಾಂಶ theta = theta_interp;
    ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
        ದಶಮಾಂಶ ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_ಗುಣಕ;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::ಲಘುಗಣಕ(1.0f / freq_scale);
    }
    cos_theta = sycl::ಕೋಜ್ಯಾ(theta) * mscale;
    sin_theta = sycl::ಜ್ಯಾ(theta) * mscale;
    ಆದರೆ (!forward) {
        sin_theta *= -1.0f;
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು D>
ಸ್ಥಿರ ಶೂನ್ಯ rope_norm(ಸ್ಥಿರಾಂಕ T *x, D *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02,
                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3,
                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs, ಸ್ಥಿರಾಂಕ int32_t *pos,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors,
                      ಸ್ಥಿರಾಂಕ int64_t *ಅಡ್ಡಸಾಲು_indices, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = 2 * (ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                        ಅಂಶ_ct1.get_local_id(1));

    ಆದರೆ (i0 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                        ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ uint32_t i3 = ಅಡ್ಡಸಾಲು_dst / (ne01 * ne02);
    ಸ್ಥಿರಾಂಕ uint32_t i2 = (ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02) / ne01;
    ಸ್ಥಿರಾಂಕ uint32_t i1 = ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಂಕ idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride != 0) {
        idst = i1 * s1 + i0;
        idst += ಅಡ್ಡಸಾಲು_indices[i2] * ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride;
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &store_coaelsced = [&](ದಶಮಾಂಶ x0, ದಶಮಾಂಶ x1) {
        ಆದರೆ constexpr (std::is_same_v<ದಶಮಾಂಶ, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    ಆದರೆ (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iw = i0 - n_offs; // relative idx

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_ತಳ = pos[i2] * dpct::ಘಾತ(theta_scale, iw / 2.0f);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ಗುಣಕ = has_ff ? freq_factors[iw / 2] : 1.0f;

    ದಶಮಾಂಶ cos_theta;
    ದಶಮಾಂಶ sin_theta;

    rope_yarn<forward>(theta_ತಳ / freq_ಗುಣಕ, freq_scale, corr_dims, iw,
                       ext_ಗುಣಕ, attn_ಗುಣಕ, cos_theta, sin_theta);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[ix + 0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು D>
ಸ್ಥಿರ ಶೂನ್ಯ rope_neox(ಸ್ಥಿರಾಂಕ T *x, D *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02,
                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3,
                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs, ಸ್ಥಿರಾಂಕ int32_t *pos,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors,
                      ಸ್ಥಿರಾಂಕ int64_t *ಅಡ್ಡಸಾಲು_indices, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = 2 * (ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                        ಅಂಶ_ct1.get_local_id(1));

    ಆದರೆ (i0 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                        ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ uint32_t i3 = ಅಡ್ಡಸಾಲು_dst / (ne01 * ne02);
    ಸ್ಥಿರಾಂಕ uint32_t i2 = (ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02) / ne01;
    ಸ್ಥಿರಾಂಕ uint32_t i1 = ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಂಕ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += ಅಡ್ಡಸಾಲು_indices[i2] * ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride;
    }

    ಆದರೆ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iw = i0 - n_offs; // relative idx

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_ತಳ = pos[i2] * dpct::ಘಾತ(theta_scale, iw / 2.0f);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ಗುಣಕ = has_ff ? freq_factors[iw / 2] : 1.0f;

    ದಶಮಾಂಶ cos_theta;
    ದಶಮಾಂಶ sin_theta;

    rope_yarn<forward>(theta_ತಳ / freq_ಗುಣಕ, freq_scale, corr_dims, iw,
                       ext_ಗುಣಕ, attn_ಗುಣಕ, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[ix + n_offs / 2 + 0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ rope_multi(ಸ್ಥಿರಾಂಕ T *x, T *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02,
                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3,
                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs, ಸ್ಥಿರಾಂಕ int32_t *pos,
                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors,
                       ಸ್ಥಿರಾಂಕ mrope_sections sections, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = 2 * (ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                        ಅಂಶ_ct1.get_local_id(1));

    ಆದರೆ (i0 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                        ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ uint32_t i3 = ಅಡ್ಡಸಾಲು_dst / (ne01 * ne02);
    ಸ್ಥಿರಾಂಕ uint32_t i2 = (ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02) / ne01;
    ಸ್ಥಿರಾಂಕ uint32_t i1 = ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಂಕ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ಆದರೆ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iw = i0 - n_offs; // relative idx

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sec_w = sections.v[1] + sections.v[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sector = (iw / 2) % sect_dims;

    ದಶಮಾಂಶ theta_ತಳ = 0.0;
    ಆದರೆ (is_imrope) {
        ಆದರೆ (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_ತಳ = pos[i2 + ne02 * 1] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_ತಳ = pos[i2 + ne02 * 2] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_ತಳ = pos[i2] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        } ಇಲ್ಲದಿದ್ದರೆ {
            theta_ತಳ = pos[i2 + ne02 * 3] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (sector < sections.v[0]) {
            theta_ತಳ = pos[i2] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sections.v[0] && sector < sec_w) {
            theta_ತಳ = pos[i2 + ne02 * 1] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_ತಳ = pos[i2 + ne02 * 2] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_w + sections.v[2]) {
            theta_ತಳ = pos[i2 + ne02 * 3] * dpct::ಘಾತ(theta_scale, iw / 2.0f);
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ಗುಣಕ = has_ff ? freq_factors[iw / 2] : 1.0f;

    ದಶಮಾಂಶ cos_theta;
    ದಶಮಾಂಶ sin_theta;

    rope_yarn<forward>(theta_ತಳ / freq_ಗುಣಕ, freq_scale, corr_dims, iw,
                       ext_ಗುಣಕ, attn_ಗುಣಕ, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[ix + n_offs / 2 + 0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ತಾರ್ಕಿಕ has_ff, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ rope_vision(ಸ್ಥಿರಾಂಕ T *x, T *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02,
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3,
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims, ಸ್ಥಿರಾಂಕ int32_t *pos,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors,
                        ಸ್ಥಿರಾಂಕ mrope_sections sections) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = 2 * (ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                        ಅಂಶ_ct1.get_local_id(1));

    ಆದರೆ (i0 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                        ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ uint32_t i3 = ಅಡ್ಡಸಾಲು_dst / (ne01 * ne02);
    ಸ್ಥಿರಾಂಕ uint32_t i2 = (ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02) / ne01;
    ಸ್ಥಿರಾಂಕ uint32_t i1 = ಅಡ್ಡಸಾಲು_dst - i3 * ne01 * ne02 - i2 * ne01;

    ಪೂರ್ಣಾಂಕ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sect_dims = sections.v[0] + sections.v[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sec_w = sections.v[1] + sections.v[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sector = (i0 / 2) % sect_dims;

    ದಶಮಾಂಶ theta_ತಳ = 0.0;
    ಆದರೆ (sector < sections.v[0]) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p = sector;
        theta_ತಳ = pos[i2] * dpct::ಘಾತ(theta_scale, p);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sections.v[0] && sector < sec_w) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p = sector - sections.v[0];
        theta_ತಳ = pos[i2 + ne02] * dpct::ಘಾತ(theta_scale, p);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ಗುಣಕ = has_ff ? freq_factors[i0 / 2] : 1.0f;

    ದಶಮಾಂಶ cos_theta;
    ದಶಮಾಂಶ sin_theta;

    rope_yarn<forward>(theta_ತಳ / freq_ಗುಣಕ, freq_scale, corr_dims, i0,
                       ext_ಗುಣಕ, attn_ಗುಣಕ, cos_theta, sin_theta);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = x[ix + 0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು D>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_norm_sycl(ಸ್ಥಿರಾಂಕ T *x, D *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr, ಸ್ಥಿರಾಂಕ int32_t *pos, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale,
               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors, ಸ್ಥಿರಾಂಕ int64_t *ಅಡ್ಡಸಾಲು_indices,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(1, SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ, 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು_x =
        (ne00 + 2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ - 1) / (2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(nr, n_ಖಂಡಗಳು_x, 1);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ಆದರೆ (freq_factors == ಶೂನ್ಯಸೂಚಕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_norm<forward, ಸುಳ್ಳು>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride);
            });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_norm<forward, ನಿಜ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride);
            });
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು D>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_neox_sycl(ಸ್ಥಿರಾಂಕ T *x, D *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr, ಸ್ಥಿರಾಂಕ int32_t *pos, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale,
               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors, ಸ್ಥಿರಾಂಕ int64_t *ಅಡ್ಡಸಾಲು_indices,
               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(1, SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ, 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು_x =
        (ne00 + 2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ - 1) / (2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(nr, n_ಖಂಡಗಳು_x, 1);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ಆದರೆ (freq_factors == ಶೂನ್ಯಸೂಚಕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_neox<forward, ಸುಳ್ಳು>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride);
            });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_neox<forward, ನಿಜ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride);
            });
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_multi_sycl(ಸ್ಥಿರಾಂಕ T *x, T *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr, ಸ್ಥಿರಾಂಕ int32_t *pos, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale,
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors, ಸ್ಥಿರಾಂಕ mrope_sections sections,
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(1, SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ, 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು_x =
        (ne00 + 2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ - 1) / (2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(nr, n_ಖಂಡಗಳು_x, 1);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ಆದರೆ (freq_factors == ಶೂನ್ಯಸೂಚಕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_multi<forward, ಸುಳ್ಳು, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_multi<forward, ನಿಜ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ
rope_vision_sycl(ಸ್ಥಿರಾಂಕ T *x, T *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s03,
                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims,
                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr, ಸ್ಥಿರಾಂಕ int32_t *pos, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale,
                 ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ,
                 ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ, ಸ್ಥಿರಾಂಕ rope_corr_dims corr_dims,
                 ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors, ಸ್ಥಿರಾಂಕ mrope_sections sections,
                 dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(1, SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ, 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು_x =
        (ne00 + 2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ - 1) / (2 * SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(nr, n_ಖಂಡಗಳು_x, 1);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ಆದರೆ (freq_factors == ಶೂನ್ಯಸೂಚಕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_vision<forward, ಸುಳ್ಳು, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
                rope_vision<forward, ನಿಜ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ forward>
ಶೂನ್ಯ ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ಸ್ಥಿರಾಂಕ ggml_tensor *ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = ಶೂನ್ಯಸೂಚಕ) {
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor *src2 = dst->src[2];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src1->ದತ್ತಾಂಶ;

    ಶೂನ್ಯ *dst_d = dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t *ಅಡ್ಡಸಾಲು_indices = ಶೂನ್ಯಸೂಚಕ;
    ggml_type dst_type = dst->type;
    ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride = 0;

    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು != ಶೂನ್ಯಸೂಚಕ) {
        GGML_ASSERT(forward);
        dst_d = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->ದತ್ತಾಂಶ;
        ಅಡ್ಡಸಾಲು_indices = (ಸ್ಥಿರಾಂಕ int64_t *)ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]->ದತ್ತಾಂಶ;
        dst_type = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type;
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->nb[1] / ggml_type_ಗಾತ್ರ(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type);
    }
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0]; // head dims
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1]; // num heads
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2]; // num heads
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0);

    ಸ್ಥಿರಾಂಕ size_t s01 = src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t s02 = src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t s03 = src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type);

    ಸ್ಥಿರಾಂಕ size_t s1 = dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type);
    ಸ್ಥಿರಾಂಕ size_t s2 = dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type);
    ಸ್ಥಿರಾಂಕ size_t s3 = dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims = ((int32_t *)dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ((int32_t *)dst->op_params)[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    ದಶಮಾಂಶ freq_ತಳ;
    ದಶಮಾಂಶ freq_scale;
    ದಶಮಾಂಶ ext_ಗುಣಕ;
    ದಶಮಾಂಶ attn_ಗುಣಕ;
    ದಶಮಾಂಶ beta_fast;
    ದಶಮಾಂಶ beta_slow;

    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ, (int32_t *)dst->op_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&freq_scale, (int32_t *)dst->op_params + 6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ, (int32_t *)dst->op_params + 7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, (int32_t *)dst->op_params + 8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_fast, (int32_t *)dst->op_params + 9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_slow, (int32_t *)dst->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&sections.v, (int32_t *)dst->op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * 4);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ಆದರೆ (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    ಆದರೆ (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ಸ್ಥಿರಾಂಕ int32_t *pos = (ಸ್ಥಿರಾಂಕ int32_t *)src1_d;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *freq_factors = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (src2 != ಶೂನ್ಯಸೂಚಕ) {
        freq_factors = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src2->ದತ್ತಾಂಶ;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    ಆದರೆ (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0_d, (ದಶಮಾಂಶ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_ತಳ,
                ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors, ಅಡ್ಡಸಾಲು_indices,
                ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, ದಶಮಾಂಶ, sycl::half>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors,
                ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ಸ್ಥಿರಾಂಕ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors,
                ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        ಆದರೆ (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0_d, (ದಶಮಾಂಶ *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_ತಳ,
                                     ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ಸ್ಥಿರಾಂಕ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        ಆದರೆ (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0_d, (ದಶಮಾಂಶ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_ತಳ,
                ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors, sections,
                stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ಸ್ಥಿರಾಂಕ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors,
                sections, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0_d, (ದಶಮಾಂಶ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_ತಳ,
                ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors, ಅಡ್ಡಸಾಲು_indices,
                ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, ದಶಮಾಂಶ, sycl::half>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors,
                ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ಸ್ಥಿರಾಂಕ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, corr_dims, freq_factors,
                ಅಡ್ಡಸಾಲು_indices, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

ಶೂನ್ಯ ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<ನಿಜ>(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_rope_ಹಿಂಭಾಗ(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ಸುಳ್ಳು>(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ನಿಜ>(ctx, rope, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು);
}
