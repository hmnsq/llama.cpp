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

स्थिर भिन्न rope_yarn_ramp(अचर भिन्न तल्लो, अचर भिन्न माथिल्लो, अचर पूर्णाङ्क i0) {
    अचर भिन्न y = (i0 / 2 - तल्लो) / sycl::max(0.001f, माथिल्लो - तल्लो);
    फर्काऊ 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

ढाँचा <तार्किक forward>
स्थिर शून्य rope_yarn(अचर भिन्न theta_extrap, अचर भिन्न freq_scale,
                      अचर rope_corr_dims corr_dims, अचर int64_t i0,
                      अचर भिन्न ext_factor, भिन्न mscale, भिन्न &cos_theta,
                      भिन्न &sin_theta) {
    भिन्न theta_interp = freq_scale * theta_extrap;
    भिन्न theta = theta_interp;
    यदि (ext_factor != 0.0f) {
        भिन्न ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    यदि (!forward) {
        sin_theta *= -1.0f;
    }
}

ढाँचा <तार्किक forward, तार्किक has_ff, typename T, typename D>
स्थिर शून्य rope_norm(अचर T *x, D *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
                      अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02,
                      अचर पूर्णाङ्क s03, अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3,
                      अचर पूर्णाङ्क n_dims, अचर पूर्णाङ्क n_offs, अचर int32_t *pos,
                      अचर भिन्न freq_scale, अचर भिन्न ext_factor,
                      अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
                      अचर भिन्न theta_scale, अचर भिन्न *freq_factors,
                      अचर int64_t *row_indices, अचर पूर्णाङ्क set_rows_stride) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    अचर uint32_t i3 = row_dst / (ne01 * ne02);
    अचर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    अचर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    अचर पूर्णाङ्क ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    यदि (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    अचर स्वतः &store_coaelsced = [&](भिन्न x0, भिन्न x1) {
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
        फर्काऊ;
    }

    अचर पूर्णाङ्क iw = i0 - n_offs; // relative idx

    अचर भिन्न theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);

    अचर भिन्न freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    अचर भिन्न x0 = x[ix + 0];
    अचर भिन्न x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

ढाँचा <तार्किक forward, तार्किक has_ff, typename T, typename D>
स्थिर शून्य rope_neox(अचर T *x, D *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
                      अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02,
                      अचर पूर्णाङ्क s03, अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3,
                      अचर पूर्णाङ्क n_dims, अचर पूर्णाङ्क n_offs, अचर int32_t *pos,
                      अचर भिन्न freq_scale, अचर भिन्न ext_factor,
                      अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
                      अचर भिन्न theta_scale, अचर भिन्न *freq_factors,
                      अचर int64_t *row_indices, अचर पूर्णाङ्क set_rows_stride) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    अचर uint32_t i3 = row_dst / (ne01 * ne02);
    अचर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    अचर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    अचर पूर्णाङ्क ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    यदि (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    यदि (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        फर्काऊ;
    }

    अचर पूर्णाङ्क iw = i0 - n_offs; // relative idx

    अचर भिन्न theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);

    अचर भिन्न freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    अचर भिन्न x0 = x[ix + n_offs / 2 + 0];
    अचर भिन्न x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

ढाँचा <तार्किक forward, तार्किक has_ff, typename T>
स्थिर शून्य rope_multi(अचर T *x, T *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
                       अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02,
                       अचर पूर्णाङ्क s03, अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3,
                       अचर पूर्णाङ्क n_dims, अचर पूर्णाङ्क n_offs, अचर int32_t *pos,
                       अचर भिन्न freq_scale, अचर भिन्न ext_factor,
                       अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
                       अचर भिन्न theta_scale, अचर भिन्न *freq_factors,
                       अचर mrope_sections sections, अचर तार्किक is_imrope) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    अचर uint32_t i3 = row_dst / (ne01 * ne02);
    अचर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    अचर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    अचर पूर्णाङ्क ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    यदि (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        फर्काऊ;
    }

    अचर पूर्णाङ्क iw = i0 - n_offs; // relative idx

    अचर पूर्णाङ्क sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    अचर पूर्णाङ्क sec_w = sections.v[1] + sections.v[0];
    अचर पूर्णाङ्क sector = (iw / 2) % sect_dims;

    भिन्न theta_base = 0.0;
    यदि (is_imrope) {
        यदि (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा {
            theta_base = pos[i2 + ne02 * 3] * dpct::घात(theta_scale, iw / 2.0f);
        }
    } अन्यथा {
        यदि (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा यदि (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::घात(theta_scale, iw / 2.0f);
        }
    }

    अचर भिन्न freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    अचर भिन्न x0 = x[ix + n_offs / 2 + 0];
    अचर भिन्न x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

ढाँचा <तार्किक forward, तार्किक has_ff, typename T>
स्थिर शून्य rope_vision(अचर T *x, T *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
                        अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02,
                        अचर पूर्णाङ्क s03, अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3,
                        अचर पूर्णाङ्क n_dims, अचर int32_t *pos,
                        अचर भिन्न freq_scale, अचर भिन्न ext_factor,
                        अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
                        अचर भिन्न theta_scale, अचर भिन्न *freq_factors,
                        अचर mrope_sections sections) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर पूर्णाङ्क i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    यदि (i0 >= ne00) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    अचर uint32_t i3 = row_dst / (ne01 * ne02);
    अचर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    अचर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णाङ्क idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    अचर पूर्णाङ्क ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    अचर पूर्णाङ्क sect_dims = sections.v[0] + sections.v[1];
    अचर पूर्णाङ्क sec_w = sections.v[1] + sections.v[0];
    अचर पूर्णाङ्क sector = (i0 / 2) % sect_dims;

    भिन्न theta_base = 0.0;
    यदि (sector < sections.v[0]) {
        अचर पूर्णाङ्क p = sector;
        theta_base = pos[i2] * dpct::घात(theta_scale, p);
    } अन्यथा यदि (sector >= sections.v[0] && sector < sec_w) {
        अचर पूर्णाङ्क p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::घात(theta_scale, p);
    }

    अचर भिन्न freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    भिन्न cos_theta;
    भिन्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    अचर भिन्न x0 = x[ix + 0];
    अचर भिन्न x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

ढाँचा <तार्किक forward, typename T, typename D>
स्थिर शून्य
rope_norm_sycl(अचर T *x, D *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
               अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02, अचर पूर्णाङ्क s03,
               अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3, अचर पूर्णाङ्क n_dims,
               अचर पूर्णाङ्क n_offs, अचर पूर्णाङ्क nr, अचर int32_t *pos, अचर भिन्न freq_scale,
               अचर भिन्न freq_base, अचर भिन्न ext_factor,
               अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
               अचर भिन्न *freq_factors, अचर int64_t *row_indices,
               अचर पूर्णाङ्क set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    अचर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    अचर पूर्णाङ्क n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    अचर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    यदि (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, असत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, सत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

ढाँचा <तार्किक forward, typename T, typename D>
स्थिर शून्य
rope_neox_sycl(अचर T *x, D *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
               अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02, अचर पूर्णाङ्क s03,
               अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3, अचर पूर्णाङ्क n_dims,
               अचर पूर्णाङ्क n_offs, अचर पूर्णाङ्क nr, अचर int32_t *pos, अचर भिन्न freq_scale,
               अचर भिन्न freq_base, अचर भिन्न ext_factor,
               अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
               अचर भिन्न *freq_factors, अचर int64_t *row_indices,
               अचर पूर्णाङ्क set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    अचर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    अचर पूर्णाङ्क n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    अचर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    यदि (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, असत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, सत्य>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

ढाँचा <तार्किक forward, typename T>
स्थिर शून्य
rope_multi_sycl(अचर T *x, T *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
                अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02, अचर पूर्णाङ्क s03,
                अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3, अचर पूर्णाङ्क n_dims,
                अचर पूर्णाङ्क n_offs, अचर पूर्णाङ्क nr, अचर int32_t *pos, अचर भिन्न freq_scale,
                अचर भिन्न freq_base, अचर भिन्न ext_factor,
                अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
                अचर भिन्न *freq_factors, अचर mrope_sections sections,
                अचर तार्किक is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    अचर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    अचर पूर्णाङ्क n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    अचर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    यदि (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, असत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, सत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

ढाँचा <तार्किक forward, typename T>
स्थिर शून्य
rope_vision_sycl(अचर T *x, T *dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क ne01,
                 अचर पूर्णाङ्क ne02, अचर पूर्णाङ्क s01, अचर पूर्णाङ्क s02, अचर पूर्णाङ्क s03,
                 अचर पूर्णाङ्क s1, अचर पूर्णाङ्क s2, अचर पूर्णाङ्क s3, अचर पूर्णाङ्क n_dims,
                 अचर पूर्णाङ्क nr, अचर int32_t *pos, अचर भिन्न freq_scale,
                 अचर भिन्न freq_base, अचर भिन्न ext_factor,
                 अचर भिन्न attn_factor, अचर rope_corr_dims corr_dims,
                 अचर भिन्न *freq_factors, अचर mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    अचर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    अचर पूर्णाङ्क n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    अचर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    यदि (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, असत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, सत्य, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

ढाँचा <तार्किक forward>
शून्य ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            अचर ggml_tensor *set_rows = nullptr) {
    अचर ggml_tensor *src0 = dst->src[0];
    अचर ggml_tensor *src1 = dst->src[1];
    अचर ggml_tensor *src2 = dst->src[2];

    अचर भिन्न *src0_d = (अचर भिन्न *)src0->data;
    अचर भिन्न *src1_d = (अचर भिन्न *)src1->data;

    शून्य *dst_d = dst->data;
    अचर int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    पूर्णाङ्क set_rows_stride = 0;

    यदि (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (अचर int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    अचर int64_t ne00 = src0->ne[0]; // head dims
    अचर int64_t ne01 = src0->ne[1]; // num heads
    अचर int64_t ne02 = src0->ne[2]; // num heads
    अचर int64_t nr = ggml_nrows(src0);

    अचर माप_प्रकार s01 = src0->nb[1] / ggml_type_size(src0->type);
    अचर माप_प्रकार s02 = src0->nb[2] / ggml_type_size(src0->type);
    अचर माप_प्रकार s03 = src0->nb[3] / ggml_type_size(src0->type);

    अचर माप_प्रकार s1 = dst->nb[1] / ggml_type_size(dst->type);
    अचर माप_प्रकार s2 = dst->nb[2] / ggml_type_size(dst->type);
    अचर माप_प्रकार s3 = dst->nb[3] / ggml_type_size(dst->type);

    अचर पूर्णाङ्क n_dims = ((int32_t *)dst->op_params)[1];
    अचर पूर्णाङ्क mode = ((int32_t *)dst->op_params)[2];
    अचर पूर्णाङ्क n_ctx_orig = ((int32_t *)dst->op_params)[4];
    अचर पूर्णाङ्क n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    भिन्न freq_base;
    भिन्न freq_scale;
    भिन्न ext_factor;
    भिन्न attn_factor;
    भिन्न beta_fast;
    भिन्न beta_slow;

    स_नकल(&freq_base, (int32_t *)dst->op_params + 5, परिमाण(भिन्न));
    स_नकल(&freq_scale, (int32_t *)dst->op_params + 6, परिमाण(भिन्न));
    स_नकल(&ext_factor, (int32_t *)dst->op_params + 7, परिमाण(भिन्न));
    स_नकल(&attn_factor, (int32_t *)dst->op_params + 8, परिमाण(भिन्न));
    स_नकल(&beta_fast, (int32_t *)dst->op_params + 9, परिमाण(भिन्न));
    स_नकल(&beta_slow, (int32_t *)dst->op_params + 10, परिमाण(भिन्न));
    स_नकल(&sections.v, (int32_t *)dst->op_params + 11, परिमाण(पूर्णाङ्क) * 4);

    अचर तार्किक is_neox = mode & GGML_ROPE_TYPE_NEOX;
    अचर तार्किक is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    अचर तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    अचर तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

    यदि (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    यदि (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    अचर int32_t *pos = (अचर int32_t *)src1_d;

    अचर भिन्न *freq_factors = nullptr;
    यदि (src2 != nullptr) {
        freq_factors = (अचर भिन्न *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    यदि (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, भिन्न, भिन्न>(
                (अचर भिन्न *)src0_d, (भिन्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, भिन्न, sycl::half>(
                (अचर भिन्न *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (अचर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा यदि (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((अचर भिन्न *)src0_d, (भिन्न *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (अचर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा यदि (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (अचर भिन्न *)src0_d, (भिन्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (अचर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, भिन्न, भिन्न>(
                (अचर भिन्न *)src0_d, (भिन्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, भिन्न, sycl::half>(
                (अचर भिन्न *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा यदि (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (अचर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

शून्य ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<सत्य>(ctx, dst);
}

शून्य ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<असत्य>(ctx, dst);
}

शून्य ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<सत्य>(ctx, rope, set_rows);
}
