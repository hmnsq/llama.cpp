#ਅਨ੍ਤਰ੍ਭਾਵ "rope.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

ਬਣਤਰ rope_corr_dims {
    ਦਸ਼ਮਲਵ v[2];
};

ਬਣਤਰ mrope_sections {
    ਪੂਰਨ v[4];
};

ਸਥਿਰ ਦਸ਼ਮਲਵ rope_yarn_ramp(ਅਚਲ ਦਸ਼ਮਲਵ ਨੀਵਾਂ, ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਚਾ, ਅਚਲ ਪੂਰਨ i0) {
    ਅਚਲ ਦਸ਼ਮਲਵ y = (i0 / 2 - ਨੀਵਾਂ) / sycl::max(0.001f, ਉੱਚਾ - ਨੀਵਾਂ);
    ਵਾਪਸ 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward>
ਸਥਿਰ ਖਾਲੀ rope_yarn(ਅਚਲ ਦਸ਼ਮਲਵ theta_extrap, ਅਚਲ ਦਸ਼ਮਲਵ freq_scale,
                      ਅਚਲ rope_corr_dims corr_dims, ਅਚਲ int64_t i0,
                      ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ, ਦਸ਼ਮਲਵ mscale, ਦਸ਼ਮਲਵ &cos_theta,
                      ਦਸ਼ਮਲਵ &sin_theta) {
    ਦਸ਼ਮਲਵ theta_interp = freq_scale * theta_extrap;
    ਦਸ਼ਮਲਵ theta = theta_interp;
    ਜੇ (ext_ਗੁਣਕ != 0.0f) {
        ਦਸ਼ਮਲਵ ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_ਗੁਣਕ;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::ਲਘੂਗਣਕ(1.0f / freq_scale);
    }
    cos_theta = sycl::ਕੋਜੀਵਾ(theta) * mscale;
    sin_theta = sycl::ਜੀਵਾ(theta) * mscale;
    ਜੇ (!forward) {
        sin_theta *= -1.0f;
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਤਾਰਕਿਕ has_ff, ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ D>
ਸਥਿਰ ਖਾਲੀ rope_norm(ਅਚਲ T *x, D *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                      ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02,
                      ਅਚਲ ਪੂਰਨ s03, ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3,
                      ਅਚਲ ਪੂਰਨ n_dims, ਅਚਲ ਪੂਰਨ n_offs, ਅਚਲ int32_t *pos,
                      ਅਚਲ ਦਸ਼ਮਲਵ freq_scale, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
                      ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
                      ਅਚਲ ਦਸ਼ਮਲਵ theta_scale, ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors,
                      ਅਚਲ int64_t *ਪੰਕਤੀ_indices, ਅਚਲ ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ i0 = 2 * (ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                        ਵਸਤੂ_ct1.get_local_id(1));

    ਜੇ (i0 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                        ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ uint32_t i3 = ਪੰਕਤੀ_dst / (ne01 * ne02);
    ਅਚਲ uint32_t i2 = (ਪੰਕਤੀ_dst - i3 * ne01 * ne02) / ne01;
    ਅਚਲ uint32_t i1 = ਪੰਕਤੀ_dst - i3 * ne01 * ne02 - i2 * ne01;

    ਪੂਰਨ idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ਅਚਲ ਪੂਰਨ ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride != 0) {
        idst = i1 * s1 + i0;
        idst += ਪੰਕਤੀ_indices[i2] * ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride;
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ &store_coaelsced = [&](ਦਸ਼ਮਲਵ x0, ਦਸ਼ਮਲਵ x1) {
        ਜੇ constexpr (std::is_same_v<ਦਸ਼ਮਲਵ, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    ਜੇ (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ iw = i0 - n_offs; // relative idx

    ਅਚਲ ਦਸ਼ਮਲਵ theta_ਅਧਾਰ = pos[i2] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);

    ਅਚਲ ਦਸ਼ਮਲਵ freq_ਗੁਣਕ = has_ff ? freq_factors[iw / 2] : 1.0f;

    ਦਸ਼ਮਲਵ cos_theta;
    ਦਸ਼ਮਲਵ sin_theta;

    rope_yarn<forward>(theta_ਅਧਾਰ / freq_ਗੁਣਕ, freq_scale, corr_dims, iw,
                       ext_ਗੁਣਕ, attn_ਗੁਣਕ, cos_theta, sin_theta);

    ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[ix + 0];
    ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਤਾਰਕਿਕ has_ff, ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ D>
ਸਥਿਰ ਖਾਲੀ rope_neox(ਅਚਲ T *x, D *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                      ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02,
                      ਅਚਲ ਪੂਰਨ s03, ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3,
                      ਅਚਲ ਪੂਰਨ n_dims, ਅਚਲ ਪੂਰਨ n_offs, ਅਚਲ int32_t *pos,
                      ਅਚਲ ਦਸ਼ਮਲਵ freq_scale, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
                      ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
                      ਅਚਲ ਦਸ਼ਮਲਵ theta_scale, ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors,
                      ਅਚਲ int64_t *ਪੰਕਤੀ_indices, ਅਚਲ ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ i0 = 2 * (ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                        ਵਸਤੂ_ct1.get_local_id(1));

    ਜੇ (i0 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                        ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ uint32_t i3 = ਪੰਕਤੀ_dst / (ne01 * ne02);
    ਅਚਲ uint32_t i2 = (ਪੰਕਤੀ_dst - i3 * ne01 * ne02) / ne01;
    ਅਚਲ uint32_t i1 = ਪੰਕਤੀ_dst - i3 * ne01 * ne02 - i2 * ne01;

    ਪੂਰਨ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ਅਚਲ ਪੂਰਨ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += ਪੰਕਤੀ_indices[i2] * ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride;
    }

    ਜੇ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ iw = i0 - n_offs; // relative idx

    ਅਚਲ ਦਸ਼ਮਲਵ theta_ਅਧਾਰ = pos[i2] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);

    ਅਚਲ ਦਸ਼ਮਲਵ freq_ਗੁਣਕ = has_ff ? freq_factors[iw / 2] : 1.0f;

    ਦਸ਼ਮਲਵ cos_theta;
    ਦਸ਼ਮਲਵ sin_theta;

    rope_yarn<forward>(theta_ਅਧਾਰ / freq_ਗੁਣਕ, freq_scale, corr_dims, iw,
                       ext_ਗੁਣਕ, attn_ਗੁਣਕ, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[ix + n_offs / 2 + 0];
    ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਤਾਰਕਿਕ has_ff, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ rope_multi(ਅਚਲ T *x, T *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                       ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02,
                       ਅਚਲ ਪੂਰਨ s03, ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3,
                       ਅਚਲ ਪੂਰਨ n_dims, ਅਚਲ ਪੂਰਨ n_offs, ਅਚਲ int32_t *pos,
                       ਅਚਲ ਦਸ਼ਮਲਵ freq_scale, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
                       ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
                       ਅਚਲ ਦਸ਼ਮਲਵ theta_scale, ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors,
                       ਅਚਲ mrope_sections sections, ਅਚਲ ਤਾਰਕਿਕ is_imrope) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ i0 = 2 * (ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                        ਵਸਤੂ_ct1.get_local_id(1));

    ਜੇ (i0 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                        ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ uint32_t i3 = ਪੰਕਤੀ_dst / (ne01 * ne02);
    ਅਚਲ uint32_t i2 = (ਪੰਕਤੀ_dst - i3 * ne01 * ne02) / ne01;
    ਅਚਲ uint32_t i1 = ਪੰਕਤੀ_dst - i3 * ne01 * ne02 - i2 * ne01;

    ਪੂਰਨ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ਅਚਲ ਪੂਰਨ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ਜੇ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ iw = i0 - n_offs; // relative idx

    ਅਚਲ ਪੂਰਨ sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ਅਚਲ ਪੂਰਨ sec_w = sections.v[1] + sections.v[0];
    ਅਚਲ ਪੂਰਨ sector = (iw / 2) % sect_dims;

    ਦਸ਼ਮਲਵ theta_ਅਧਾਰ = 0.0;
    ਜੇ (is_imrope) {
        ਜੇ (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_ਅਧਾਰ = pos[i2 + ne02 * 1] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_ਅਧਾਰ = pos[i2 + ne02 * 2] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_ਅਧਾਰ = pos[i2] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        } ਨਹੀਂਤਾਂ {
            theta_ਅਧਾਰ = pos[i2 + ne02 * 3] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (sector < sections.v[0]) {
            theta_ਅਧਾਰ = pos[i2] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        } ਨਹੀਂਤਾਂ ਜੇ (sector >= sections.v[0] && sector < sec_w) {
            theta_ਅਧਾਰ = pos[i2 + ne02 * 1] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        } ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_ਅਧਾਰ = pos[i2 + ne02 * 2] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        } ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_w + sections.v[2]) {
            theta_ਅਧਾਰ = pos[i2 + ne02 * 3] * dpct::ਘਾਤ(theta_scale, iw / 2.0f);
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ freq_ਗੁਣਕ = has_ff ? freq_factors[iw / 2] : 1.0f;

    ਦਸ਼ਮਲਵ cos_theta;
    ਦਸ਼ਮਲਵ sin_theta;

    rope_yarn<forward>(theta_ਅਧਾਰ / freq_ਗੁਣਕ, freq_scale, corr_dims, iw,
                       ext_ਗੁਣਕ, attn_ਗੁਣਕ, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[ix + n_offs / 2 + 0];
    ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਤਾਰਕਿਕ has_ff, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ rope_vision(ਅਚਲ T *x, T *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                        ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02,
                        ਅਚਲ ਪੂਰਨ s03, ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3,
                        ਅਚਲ ਪੂਰਨ n_dims, ਅਚਲ int32_t *pos,
                        ਅਚਲ ਦਸ਼ਮਲਵ freq_scale, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
                        ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
                        ਅਚਲ ਦਸ਼ਮਲਵ theta_scale, ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors,
                        ਅਚਲ mrope_sections sections) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ i0 = 2 * (ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                        ਵਸਤੂ_ct1.get_local_id(1));

    ਜੇ (i0 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                        ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ uint32_t i3 = ਪੰਕਤੀ_dst / (ne01 * ne02);
    ਅਚਲ uint32_t i2 = (ਪੰਕਤੀ_dst - i3 * ne01 * ne02) / ne01;
    ਅਚਲ uint32_t i1 = ਪੰਕਤੀ_dst - i3 * ne01 * ne02 - i2 * ne01;

    ਪੂਰਨ idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ਅਚਲ ਪੂਰਨ ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ਅਚਲ ਪੂਰਨ sect_dims = sections.v[0] + sections.v[1];
    ਅਚਲ ਪੂਰਨ sec_w = sections.v[1] + sections.v[0];
    ਅਚਲ ਪੂਰਨ sector = (i0 / 2) % sect_dims;

    ਦਸ਼ਮਲਵ theta_ਅਧਾਰ = 0.0;
    ਜੇ (sector < sections.v[0]) {
        ਅਚਲ ਪੂਰਨ p = sector;
        theta_ਅਧਾਰ = pos[i2] * dpct::ਘਾਤ(theta_scale, p);
    } ਨਹੀਂਤਾਂ ਜੇ (sector >= sections.v[0] && sector < sec_w) {
        ਅਚਲ ਪੂਰਨ p = sector - sections.v[0];
        theta_ਅਧਾਰ = pos[i2 + ne02] * dpct::ਘਾਤ(theta_scale, p);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ freq_ਗੁਣਕ = has_ff ? freq_factors[i0 / 2] : 1.0f;

    ਦਸ਼ਮਲਵ cos_theta;
    ਦਸ਼ਮਲਵ sin_theta;

    rope_yarn<forward>(theta_ਅਧਾਰ / freq_ਗੁਣਕ, freq_scale, corr_dims, i0,
                       ext_ਗੁਣਕ, attn_ਗੁਣਕ, cos_theta, sin_theta);

    ਅਚਲ ਦਸ਼ਮਲਵ x0 = x[ix + 0];
    ਅਚਲ ਦਸ਼ਮਲਵ x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ D>
ਸਥਿਰ ਖਾਲੀ
rope_norm_sycl(ਅਚਲ T *x, D *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
               ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02, ਅਚਲ ਪੂਰਨ s03,
               ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3, ਅਚਲ ਪੂਰਨ n_dims,
               ਅਚਲ ਪੂਰਨ n_offs, ਅਚਲ ਪੂਰਨ nr, ਅਚਲ int32_t *pos, ਅਚਲ ਦਸ਼ਮਲਵ freq_scale,
               ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
               ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
               ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors, ਅਚਲ int64_t *ਪੰਕਤੀ_indices,
               ਅਚਲ ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ਅਚਲ dpct::dim3 ਖੰਡ_dims(1, SYCL_ROPE_ਖੰਡ_ਮਾਪ, 1);
    ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ_x =
        (ne00 + 2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ - 1) / (2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(nr, n_ਖੰਡਾਂ_x, 1);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਜੇ (freq_factors == ਖਾਲੀ_ਸੂਚਕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_norm<forward, ਝੂਠ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride);
            });
    } ਨਹੀਂਤਾਂ {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_norm<forward, ਸੱਚ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride);
            });
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ D>
ਸਥਿਰ ਖਾਲੀ
rope_neox_sycl(ਅਚਲ T *x, D *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
               ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02, ਅਚਲ ਪੂਰਨ s03,
               ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3, ਅਚਲ ਪੂਰਨ n_dims,
               ਅਚਲ ਪੂਰਨ n_offs, ਅਚਲ ਪੂਰਨ nr, ਅਚਲ int32_t *pos, ਅਚਲ ਦਸ਼ਮਲਵ freq_scale,
               ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
               ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
               ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors, ਅਚਲ int64_t *ਪੰਕਤੀ_indices,
               ਅਚਲ ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ਅਚਲ dpct::dim3 ਖੰਡ_dims(1, SYCL_ROPE_ਖੰਡ_ਮਾਪ, 1);
    ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ_x =
        (ne00 + 2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ - 1) / (2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(nr, n_ਖੰਡਾਂ_x, 1);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਜੇ (freq_factors == ਖਾਲੀ_ਸੂਚਕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_neox<forward, ਝੂਠ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride);
            });
    } ਨਹੀਂਤਾਂ {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_neox<forward, ਸੱਚ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride);
            });
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ
rope_multi_sycl(ਅਚਲ T *x, T *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02, ਅਚਲ ਪੂਰਨ s03,
                ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3, ਅਚਲ ਪੂਰਨ n_dims,
                ਅਚਲ ਪੂਰਨ n_offs, ਅਚਲ ਪੂਰਨ nr, ਅਚਲ int32_t *pos, ਅਚਲ ਦਸ਼ਮਲਵ freq_scale,
                ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
                ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
                ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors, ਅਚਲ mrope_sections sections,
                ਅਚਲ ਤਾਰਕਿਕ is_imrope, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ਅਚਲ dpct::dim3 ਖੰਡ_dims(1, SYCL_ROPE_ਖੰਡ_ਮਾਪ, 1);
    ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ_x =
        (ne00 + 2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ - 1) / (2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(nr, n_ਖੰਡਾਂ_x, 1);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਜੇ (freq_factors == ਖਾਲੀ_ਸੂਚਕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_multi<forward, ਝੂਠ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } ਨਹੀਂਤਾਂ {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_multi<forward, ਸੱਚ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ
rope_vision_sycl(ਅਚਲ T *x, T *dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                 ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ s01, ਅਚਲ ਪੂਰਨ s02, ਅਚਲ ਪੂਰਨ s03,
                 ਅਚਲ ਪੂਰਨ s1, ਅਚਲ ਪੂਰਨ s2, ਅਚਲ ਪੂਰਨ s3, ਅਚਲ ਪੂਰਨ n_dims,
                 ਅਚਲ ਪੂਰਨ nr, ਅਚਲ int32_t *pos, ਅਚਲ ਦਸ਼ਮਲਵ freq_scale,
                 ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ,
                 ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ, ਅਚਲ rope_corr_dims corr_dims,
                 ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors, ਅਚਲ mrope_sections sections,
                 dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ਅਚਲ dpct::dim3 ਖੰਡ_dims(1, SYCL_ROPE_ਖੰਡ_ਮਾਪ, 1);
    ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ_x =
        (ne00 + 2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ - 1) / (2 * SYCL_ROPE_ਖੰਡ_ਮਾਪ);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(nr, n_ਖੰਡਾਂ_x, 1);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਜੇ (freq_factors == ਖਾਲੀ_ਸੂਚਕ) {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_vision<forward, ਝੂਠ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } ਨਹੀਂਤਾਂ {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                rope_vision<forward, ਸੱਚ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ forward>
ਖਾਲੀ ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ਅਚਲ ggml_tensor *ਸੈੱਟ_ਪੰਕਤੀਆਂ = ਖਾਲੀ_ਸੂਚਕ) {
    ਅਚਲ ggml_tensor *src0 = dst->src[0];
    ਅਚਲ ggml_tensor *src1 = dst->src[1];
    ਅਚਲ ggml_tensor *src2 = dst->src[2];

    ਅਚਲ ਦਸ਼ਮਲਵ *src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src1->ਅੰਕੜੇ;

    ਖਾਲੀ *dst_d = dst->ਅੰਕੜੇ;
    ਅਚਲ int64_t *ਪੰਕਤੀ_indices = ਖਾਲੀ_ਸੂਚਕ;
    ggml_type dst_type = dst->type;
    ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride = 0;

    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_ASSERT(forward);
        dst_d = ਸੈੱਟ_ਪੰਕਤੀਆਂ->ਅੰਕੜੇ;
        ਪੰਕਤੀ_indices = (ਅਚਲ int64_t *)ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]->ਅੰਕੜੇ;
        dst_type = ਸੈੱਟ_ਪੰਕਤੀਆਂ->type;
        ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride = ਸੈੱਟ_ਪੰਕਤੀਆਂ->nb[1] / ggml_type_ਮਾਪ(ਸੈੱਟ_ਪੰਕਤੀਆਂ->type);
    }
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ਅਚਲ int64_t ne00 = src0->ne[0]; // head dims
    ਅਚਲ int64_t ne01 = src0->ne[1]; // num heads
    ਅਚਲ int64_t ne02 = src0->ne[2]; // num heads
    ਅਚਲ int64_t nr = ggml_nrows(src0);

    ਅਚਲ size_t s01 = src0->nb[1] / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t s02 = src0->nb[2] / ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t s03 = src0->nb[3] / ggml_type_ਮਾਪ(src0->type);

    ਅਚਲ size_t s1 = dst->nb[1] / ggml_type_ਮਾਪ(dst->type);
    ਅਚਲ size_t s2 = dst->nb[2] / ggml_type_ਮਾਪ(dst->type);
    ਅਚਲ size_t s3 = dst->nb[3] / ggml_type_ਮਾਪ(dst->type);

    ਅਚਲ ਪੂਰਨ n_dims = ((int32_t *)dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode = ((int32_t *)dst->op_params)[2];
    ਅਚਲ ਪੂਰਨ n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ਅਚਲ ਪੂਰਨ n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ freq_scale;
    ਦਸ਼ਮਲਵ ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ beta_fast;
    ਦਸ਼ਮਲਵ beta_slow;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ, (int32_t *)dst->op_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale, (int32_t *)dst->op_params + 6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ, (int32_t *)dst->op_params + 7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (int32_t *)dst->op_params + 8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast, (int32_t *)dst->op_params + 9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow, (int32_t *)dst->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&sections.v, (int32_t *)dst->op_params + 11, ਆਕਾਰ(ਪੂਰਨ) * 4);

    ਅਚਲ ਤਾਰਕਿਕ is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ਅਚਲ ਤਾਰਕਿਕ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ਅਚਲ ਤਾਰਕਿਕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ਅਚਲ ਤਾਰਕਿਕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ਜੇ (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    ਜੇ (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ਅਚਲ int32_t *pos = (ਅਚਲ int32_t *)src1_d;

    ਅਚਲ ਦਸ਼ਮਲਵ *freq_factors = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (src2 != ਖਾਲੀ_ਸੂਚਕ) {
        freq_factors = (ਅਚਲ ਦਸ਼ਮਲਵ *)src2->ਅੰਕੜੇ;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    ਜੇ (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        ਜੇ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *)src0_d, (ਦਸ਼ਮਲਵ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_ਅਧਾਰ,
                ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors, ਪੰਕਤੀ_indices,
                ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, ਦਸ਼ਮਲਵ, sycl::half>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors,
                ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ਅਚਲ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors,
                ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, stream);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ਨਹੀਂਤਾਂ ਜੇ (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        ਜੇ (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ਅਚਲ ਦਸ਼ਮਲਵ *)src0_d, (ਦਸ਼ਮਲਵ *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_ਅਧਾਰ,
                                     ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ਅਚਲ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ਨਹੀਂਤਾਂ ਜੇ (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        ਜੇ (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *)src0_d, (ਦਸ਼ਮਲਵ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_ਅਧਾਰ,
                ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors, sections,
                stream);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ਅਚਲ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors,
                sections, stream);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ਨਹੀਂਤਾਂ {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        ਜੇ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *)src0_d, (ਦਸ਼ਮਲਵ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_ਅਧਾਰ,
                ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors, ਪੰਕਤੀ_indices,
                ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, ਦਸ਼ਮਲਵ, sycl::half>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors,
                ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ਅਚਲ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, corr_dims, freq_factors,
                ਪੰਕਤੀ_indices, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride, stream);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

ਖਾਲੀ ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<ਸੱਚ>(ctx, dst);
}

ਖਾਲੀ ggml_sycl_rope_ਪਿੱਠ(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ਝੂਠ>(ctx, dst);
}

ਖਾਲੀ ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ਸੱਚ>(ctx, rope, ਸੈੱਟ_ਪੰਕਤੀਆਂ);
}
