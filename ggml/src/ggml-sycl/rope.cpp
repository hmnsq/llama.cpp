#अन्तर्भाव "rope.hpp"
#अन्तर्भाव "convert.hpp"
#अन्तर्भाव "ggml-sycl/common.hpp"
#अन्तर्भाव "ggml.h"

संरचना rope_corr_dims {
    भिन्न v[2];
};

संरचना mrope_sections {
    पूर्णाङ्क v[4];
};

स्थिर भिन्न rope_yarn_ramp(ध्रुव भिन्न नीचः, ध्रुव भिन्न उच्चः, ध्रुव पूर्णाङ्क i0) {
    ध्रुव भिन्न y = (i0 / 2 - नीचः) / sycl::max(0.001f, उच्चः - नीचः);
    प्रत्यागम 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

प्रतिरूपम् <तार्किक forward>
स्थिर शून्य rope_yarn(ध्रुव भिन्न theta_extrap, ध्रुव भिन्न freq_scale,
                      ध्रुव rope_corr_dims corr_dims, ध्रुव int64_t i0,
                      ध्रुव भिन्न ext_गुणकः, भिन्न mscale, भिन्न &cos_theta,
                      भिन्न &sin_theta) {
    भिन्न theta_interp = freq_scale * theta_extrap;
    भिन्न theta = theta_interp;
    यदि (ext_गुणकः != 0.0f) {
        भिन्न ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_गुणकः;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::लघुगणकः(1.0f / freq_scale);
    }
    cos_theta = sycl::कोटिज्या(theta) * mscale;
    sin_theta = sycl::ज्या(theta) * mscale;
    यदि (!forward) {
        sin_theta *= -1.0f;
    }
}

प्रतिरूपम् <तार्किक forward, तार्किक has_ff, प्रकारनाम T, प्रकारनाम D>
स्थिर शून्य rope_norm(ध्रुव T *x, D *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
                      ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02,
                      ध्रुव पूर्णाङ्क s03, ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3,
                      ध्रुव पूर्णाङ्क n_dims, ध्रुव पूर्णाङ्क n_offs, ध्रुव int32_t *pos,
                      ध्रुव भिन्न freq_scale, ध्रुव भिन्न ext_गुणकः,
                      ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
                      ध्रुव भिन्न theta_scale, ध्रुव भिन्न *freq_factors,
                      ध्रुव int64_t *row_indices, ध्रुव पूर्णाङ्क समुच्चयः_पङ्क्तिकाः_stride) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                        item_ct1.get_local_id(2);

    ध्रुव uint32_t i3 = row_dst / (ne01 * ne02);
    ध्रुव uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ध्रुव uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ध्रुव पूर्णाङ्क ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    यदि (समुच्चयः_पङ्क्तिकाः_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * समुच्चयः_पङ्क्तिकाः_stride;
    }

    ध्रुव स्वचालित &store_coaelsced = [&](भिन्न x0, भिन्न x1) {
        यदि constexpr (std::is_same_v<भिन्न, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } अन्यथा यदि constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    यदि (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क iw = i0 - n_offs; // relative idx

    ध्रुव भिन्न theta_आधारः = pos[i2] * dpct::घातः(theta_scale, iw / 2.0f);

    ध्रुव भिन्न freq_गुणकः = has_ff ? freq_factors[iw / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_आधारः / freq_गुणकः, freq_scale, corr_dims, iw,
                       ext_गुणकः, attn_गुणकः, cos_theta, sin_theta);

    ध्रुव भिन्न x0 = x[ix + 0];
    ध्रुव भिन्न x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

प्रतिरूपम् <तार्किक forward, तार्किक has_ff, प्रकारनाम T, प्रकारनाम D>
स्थिर शून्य rope_neox(ध्रुव T *x, D *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
                      ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02,
                      ध्रुव पूर्णाङ्क s03, ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3,
                      ध्रुव पूर्णाङ्क n_dims, ध्रुव पूर्णाङ्क n_offs, ध्रुव int32_t *pos,
                      ध्रुव भिन्न freq_scale, ध्रुव भिन्न ext_गुणकः,
                      ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
                      ध्रुव भिन्न theta_scale, ध्रुव भिन्न *freq_factors,
                      ध्रुव int64_t *row_indices, ध्रुव पूर्णाङ्क समुच्चयः_पङ्क्तिकाः_stride) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                        item_ct1.get_local_id(2);

    ध्रुव uint32_t i3 = row_dst / (ne01 * ne02);
    ध्रुव uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ध्रुव uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ध्रुव पूर्णाङ्क ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    यदि (समुच्चयः_पङ्क्तिकाः_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * समुच्चयः_पङ्क्तिकाः_stride;
    }

    यदि (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क iw = i0 - n_offs; // relative idx

    ध्रुव भिन्न theta_आधारः = pos[i2] * dpct::घातः(theta_scale, iw / 2.0f);

    ध्रुव भिन्न freq_गुणकः = has_ff ? freq_factors[iw / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_आधारः / freq_गुणकः, freq_scale, corr_dims, iw,
                       ext_गुणकः, attn_गुणकः, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ध्रुव भिन्न x0 = x[ix + n_offs / 2 + 0];
    ध्रुव भिन्न x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

प्रतिरूपम् <तार्किक forward, तार्किक has_ff, प्रकारनाम T>
स्थिर शून्य rope_multi(ध्रुव T *x, T *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
                       ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02,
                       ध्रुव पूर्णाङ्क s03, ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3,
                       ध्रुव पूर्णाङ्क n_dims, ध्रुव पूर्णाङ्क n_offs, ध्रुव int32_t *pos,
                       ध्रुव भिन्न freq_scale, ध्रुव भिन्न ext_गुणकः,
                       ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
                       ध्रुव भिन्न theta_scale, ध्रुव भिन्न *freq_factors,
                       ध्रुव mrope_sections sections, ध्रुव तार्किक is_imrope) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                        item_ct1.get_local_id(2);

    ध्रुव uint32_t i3 = row_dst / (ne01 * ne02);
    ध्रुव uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ध्रुव uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ध्रुव पूर्णाङ्क ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    यदि (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क iw = i0 - n_offs; // relative idx

    ध्रुव पूर्णाङ्क sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ध्रुव पूर्णाङ्क sec_w = sections.v[1] + sections.v[0];
    ध्रुव पूर्णाङ्क sector = (iw / 2) % sect_dims;

    भिन्न theta_आधारः = 0.0;
    यदि (is_imrope) {
        यदि (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_आधारः = pos[i2 + ne02 * 1] * dpct::घातः(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_आधारः = pos[i2 + ne02 * 2] * dpct::घातः(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_आधारः = pos[i2] * dpct::घातः(theta_scale, iw / 2.0f);
        } अन्यथा {
            theta_आधारः = pos[i2 + ne02 * 3] * dpct::घातः(theta_scale, iw / 2.0f);
        }
    } अन्यथा {
        यदि (sector < sections.v[0]) {
            theta_आधारः = pos[i2] * dpct::घातः(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector >= sections.v[0] && sector < sec_w) {
            theta_आधारः = pos[i2 + ne02 * 1] * dpct::घातः(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_आधारः = pos[i2 + ne02 * 2] * dpct::घातः(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector >= sec_w + sections.v[2]) {
            theta_आधारः = pos[i2 + ne02 * 3] * dpct::घातः(theta_scale, iw / 2.0f);
        }
    }

    ध्रुव भिन्न freq_गुणकः = has_ff ? freq_factors[iw / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_आधारः / freq_गुणकः, freq_scale, corr_dims, iw,
                       ext_गुणकः, attn_गुणकः, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ध्रुव भिन्न x0 = x[ix + n_offs / 2 + 0];
    ध्रुव भिन्न x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

प्रतिरूपम् <तार्किक forward, तार्किक has_ff, प्रकारनाम T>
स्थिर शून्य rope_vision(ध्रुव T *x, T *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
                        ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02,
                        ध्रुव पूर्णाङ्क s03, ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3,
                        ध्रुव पूर्णाङ्क n_dims, ध्रुव int32_t *pos,
                        ध्रुव भिन्न freq_scale, ध्रुव भिन्न ext_गुणकः,
                        ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
                        ध्रुव भिन्न theta_scale, ध्रुव भिन्न *freq_factors,
                        ध्रुव mrope_sections sections) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                        item_ct1.get_local_id(2);

    ध्रुव uint32_t i3 = row_dst / (ne01 * ne02);
    ध्रुव uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ध्रुव uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ध्रुव पूर्णाङ्क ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ध्रुव पूर्णाङ्क sect_dims = sections.v[0] + sections.v[1];
    ध्रुव पूर्णाङ्क sec_w = sections.v[1] + sections.v[0];
    ध्रुव पूर्णाङ्क sector = (i0 / 2) % sect_dims;

    भिन्न theta_आधारः = 0.0;
    यदि (sector < sections.v[0]) {
        ध्रुव पूर्णाङ्क p = sector;
        theta_आधारः = pos[i2] * dpct::घातः(theta_scale, p);
    } अन्यथा यदि (sector >= sections.v[0] && sector < sec_w) {
        ध्रुव पूर्णाङ्क p = sector - sections.v[0];
        theta_आधारः = pos[i2 + ne02] * dpct::घातः(theta_scale, p);
    }

    ध्रुव भिन्न freq_गुणकः = has_ff ? freq_factors[i0 / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_आधारः / freq_गुणकः, freq_scale, corr_dims, i0,
                       ext_गुणकः, attn_गुणकः, cos_theta, sin_theta);

    ध्रुव भिन्न x0 = x[ix + 0];
    ध्रुव भिन्न x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

प्रतिरूपम् <तार्किक forward, प्रकारनाम T, प्रकारनाम D>
स्थिर शून्य
rope_norm_sycl(ध्रुव T *x, D *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
               ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02, ध्रुव पूर्णाङ्क s03,
               ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3, ध्रुव पूर्णाङ्क n_dims,
               ध्रुव पूर्णाङ्क n_offs, ध्रुव पूर्णाङ्क nr, ध्रुव int32_t *pos, ध्रुव भिन्न freq_scale,
               ध्रुव भिन्न freq_आधारः, ध्रुव भिन्न ext_गुणकः,
               ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
               ध्रुव भिन्न *freq_factors, ध्रुव int64_t *row_indices,
               ध्रुव पूर्णाङ्क समुच्चयः_पङ्क्तिकाः_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ध्रुव dpct::dim3 खण्डम्_dims(1, SYCL_ROPE_खण्डम्_आकार, 1);
    ध्रुव पूर्णाङ्क n_खण्डानि_x =
        (ne00 + 2 * SYCL_ROPE_खण्डम्_आकार - 1) / (2 * SYCL_ROPE_खण्डम्_आकार);
    ध्रुव dpct::dim3 खण्डम्_nums(nr, n_खण्डानि_x, 1);

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f / n_dims);

    यदि (freq_factors == शून्यसूचकः) {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_norm<forward, असत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, row_indices, समुच्चयः_पङ्क्तिकाः_stride);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_norm<forward, सत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, row_indices, समुच्चयः_पङ्क्तिकाः_stride);
            });
    }
}

प्रतिरूपम् <तार्किक forward, प्रकारनाम T, प्रकारनाम D>
स्थिर शून्य
rope_neox_sycl(ध्रुव T *x, D *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
               ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02, ध्रुव पूर्णाङ्क s03,
               ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3, ध्रुव पूर्णाङ्क n_dims,
               ध्रुव पूर्णाङ्क n_offs, ध्रुव पूर्णाङ्क nr, ध्रुव int32_t *pos, ध्रुव भिन्न freq_scale,
               ध्रुव भिन्न freq_आधारः, ध्रुव भिन्न ext_गुणकः,
               ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
               ध्रुव भिन्न *freq_factors, ध्रुव int64_t *row_indices,
               ध्रुव पूर्णाङ्क समुच्चयः_पङ्क्तिकाः_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ध्रुव dpct::dim3 खण्डम्_dims(1, SYCL_ROPE_खण्डम्_आकार, 1);
    ध्रुव पूर्णाङ्क n_खण्डानि_x =
        (ne00 + 2 * SYCL_ROPE_खण्डम्_आकार - 1) / (2 * SYCL_ROPE_खण्डम्_आकार);
    ध्रुव dpct::dim3 खण्डम्_nums(nr, n_खण्डानि_x, 1);

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f / n_dims);

    यदि (freq_factors == शून्यसूचकः) {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_neox<forward, असत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, row_indices, समुच्चयः_पङ्क्तिकाः_stride);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_neox<forward, सत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, row_indices, समुच्चयः_पङ्क्तिकाः_stride);
            });
    }
}

प्रतिरूपम् <तार्किक forward, प्रकारनाम T>
स्थिर शून्य
rope_multi_sycl(ध्रुव T *x, T *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
                ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02, ध्रुव पूर्णाङ्क s03,
                ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3, ध्रुव पूर्णाङ्क n_dims,
                ध्रुव पूर्णाङ्क n_offs, ध्रुव पूर्णाङ्क nr, ध्रुव int32_t *pos, ध्रुव भिन्न freq_scale,
                ध्रुव भिन्न freq_आधारः, ध्रुव भिन्न ext_गुणकः,
                ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
                ध्रुव भिन्न *freq_factors, ध्रुव mrope_sections sections,
                ध्रुव तार्किक is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ध्रुव dpct::dim3 खण्डम्_dims(1, SYCL_ROPE_खण्डम्_आकार, 1);
    ध्रुव पूर्णाङ्क n_खण्डानि_x =
        (ne00 + 2 * SYCL_ROPE_खण्डम्_आकार - 1) / (2 * SYCL_ROPE_खण्डम्_आकार);
    ध्रुव dpct::dim3 खण्डम्_nums(nr, n_खण्डानि_x, 1);

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f / n_dims);

    यदि (freq_factors == शून्यसूचकः) {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_multi<forward, असत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_multi<forward, सत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

प्रतिरूपम् <तार्किक forward, प्रकारनाम T>
स्थिर शून्य
rope_vision_sycl(ध्रुव T *x, T *dst, ध्रुव पूर्णाङ्क ne00, ध्रुव पूर्णाङ्क ne01,
                 ध्रुव पूर्णाङ्क ne02, ध्रुव पूर्णाङ्क s01, ध्रुव पूर्णाङ्क s02, ध्रुव पूर्णाङ्क s03,
                 ध्रुव पूर्णाङ्क s1, ध्रुव पूर्णाङ्क s2, ध्रुव पूर्णाङ्क s3, ध्रुव पूर्णाङ्क n_dims,
                 ध्रुव पूर्णाङ्क nr, ध्रुव int32_t *pos, ध्रुव भिन्न freq_scale,
                 ध्रुव भिन्न freq_आधारः, ध्रुव भिन्न ext_गुणकः,
                 ध्रुव भिन्न attn_गुणकः, ध्रुव rope_corr_dims corr_dims,
                 ध्रुव भिन्न *freq_factors, ध्रुव mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ध्रुव dpct::dim3 खण्डम्_dims(1, SYCL_ROPE_खण्डम्_आकार, 1);
    ध्रुव पूर्णाङ्क n_खण्डानि_x =
        (ne00 + 2 * SYCL_ROPE_खण्डम्_आकार - 1) / (2 * SYCL_ROPE_खण्डम्_आकार);
    ध्रुव dpct::dim3 खण्डम्_nums(nr, n_खण्डानि_x, 1);

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f / n_dims);

    यदि (freq_factors == शून्यसूचकः) {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_vision<forward, असत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_अप्रयुक्तम्(item_ct1);
                rope_vision<forward, सत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_गुणकः, attn_गुणकः, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

प्रतिरूपम् <तार्किक forward>
शून्य ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ध्रुव ggml_tensor *समुच्चयः_पङ्क्तिकाः = शून्यसूचकः) {
    ध्रुव ggml_tensor *src0 = dst->src[0];
    ध्रुव ggml_tensor *src1 = dst->src[1];
    ध्रुव ggml_tensor *src2 = dst->src[2];

    ध्रुव भिन्न *src0_d = (ध्रुव भिन्न *)src0->दत्त;
    ध्रुव भिन्न *src1_d = (ध्रुव भिन्न *)src1->दत्त;

    शून्य *dst_d = dst->दत्त;
    ध्रुव int64_t *row_indices = शून्यसूचकः;
    ggml_type dst_type = dst->type;
    पूर्णाङ्क समुच्चयः_पङ्क्तिकाः_stride = 0;

    यदि (समुच्चयः_पङ्क्तिकाः != शून्यसूचकः) {
        GGML_ASSERT(forward);
        dst_d = समुच्चयः_पङ्क्तिकाः->दत्त;
        row_indices = (ध्रुव int64_t *)समुच्चयः_पङ्क्तिकाः->src[1]->दत्त;
        dst_type = समुच्चयः_पङ्क्तिकाः->type;
        समुच्चयः_पङ्क्तिकाः_stride = समुच्चयः_पङ्क्तिकाः->nb[1] / ggml_type_आकार(समुच्चयः_पङ्क्तिकाः->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ध्रुव int64_t ne00 = src0->ne[0]; // head dims
    ध्रुव int64_t ne01 = src0->ne[1]; // num heads
    ध्रुव int64_t ne02 = src0->ne[2]; // num heads
    ध्रुव int64_t nr = ggml_nrows(src0);

    ध्रुव size_t s01 = src0->nb[1] / ggml_type_आकार(src0->type);
    ध्रुव size_t s02 = src0->nb[2] / ggml_type_आकार(src0->type);
    ध्रुव size_t s03 = src0->nb[3] / ggml_type_आकार(src0->type);

    ध्रुव size_t s1 = dst->nb[1] / ggml_type_आकार(dst->type);
    ध्रुव size_t s2 = dst->nb[2] / ggml_type_आकार(dst->type);
    ध्रुव size_t s3 = dst->nb[3] / ggml_type_आकार(dst->type);

    ध्रुव पूर्णाङ्क n_dims = ((int32_t *)dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode = ((int32_t *)dst->op_params)[2];
    ध्रुव पूर्णाङ्क n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ध्रुव पूर्णाङ्क n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    भिन्न freq_आधारः;
    भिन्न freq_scale;
    भिन्न ext_गुणकः;
    भिन्न attn_गुणकः;
    भिन्न beta_fast;
    भिन्न beta_slow;

    स्मृतिं_प्रतिलिख(&freq_आधारः, (int32_t *)dst->op_params + 5, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&freq_scale, (int32_t *)dst->op_params + 6, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&ext_गुणकः, (int32_t *)dst->op_params + 7, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&attn_गुणकः, (int32_t *)dst->op_params + 8, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_fast, (int32_t *)dst->op_params + 9, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_slow, (int32_t *)dst->op_params + 10, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&sections.v, (int32_t *)dst->op_params + 11, परिमाण(पूर्णाङ्क) * 4);

    ध्रुव तार्किक is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ध्रुव तार्किक is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ध्रुव तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ध्रुव तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

    यदि (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    यदि (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ध्रुव int32_t *pos = (ध्रुव int32_t *)src1_d;

    ध्रुव भिन्न *freq_factors = शून्यसूचकः;
    यदि (src2 != शून्यसूचकः) {
        freq_factors = (ध्रुव भिन्न *)src2->दत्त;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    यदि (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, भिन्न, भिन्न>(
                (ध्रुव भिन्न *)src0_d, (भिन्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_आधारः,
                ext_गुणकः, attn_गुणकः, corr_dims, freq_factors, row_indices,
                समुच्चयः_पङ्क्तिकाः_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, भिन्न, sycl::half>(
                (ध्रुव भिन्न *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_आधारः, ext_गुणकः, attn_गुणकः, corr_dims, freq_factors,
                row_indices, समुच्चयः_पङ्क्तिकाः_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ध्रुव sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_आधारः, ext_गुणकः, attn_गुणकः, corr_dims, freq_factors,
                row_indices, समुच्चयः_पङ्क्तिकाः_stride, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा यदि (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ध्रुव भिन्न *)src0_d, (भिन्न *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_आधारः,
                                     ext_गुणकः, attn_गुणकः, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ध्रुव sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_आधारः, ext_गुणकः, attn_गुणकः, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा यदि (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ध्रुव भिन्न *)src0_d, (भिन्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_आधारः,
                ext_गुणकः, attn_गुणकः, corr_dims, freq_factors, sections,
                stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ध्रुव sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_आधारः, ext_गुणकः, attn_गुणकः, corr_dims, freq_factors,
                sections, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, भिन्न, भिन्न>(
                (ध्रुव भिन्न *)src0_d, (भिन्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_आधारः,
                ext_गुणकः, attn_गुणकः, corr_dims, freq_factors, row_indices,
                समुच्चयः_पङ्क्तिकाः_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, भिन्न, sycl::half>(
                (ध्रुव भिन्न *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_आधारः, ext_गुणकः, attn_गुणकः, corr_dims, freq_factors,
                row_indices, समुच्चयः_पङ्क्तिकाः_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ध्रुव sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_आधारः, ext_गुणकः, attn_गुणकः, corr_dims, freq_factors,
                row_indices, समुच्चयः_पङ्क्तिकाः_stride, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

शून्य ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<सत्य>(ctx, dst);
}

शून्य ggml_sycl_rope_पृष्ठभागः(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<असत्य>(ctx, dst);
}

शून्य ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *समुच्चयः_पङ्क्तिकाः) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<सत्य>(ctx, rope, समुच्चयः_पङ्क्तिकाः);
}
