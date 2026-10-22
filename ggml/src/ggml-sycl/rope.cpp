#समावेश "rope.hpp"
#समावेश "convert.hpp"
#समावेश "ggml-sycl/common.hpp"
#समावेश "ggml.h"

काष्ठा rope_corr_dims {
    भग्न v[2];
};

काष्ठा mrope_sections {
    पूर्णांक v[4];
};

अटल भग्न rope_yarn_ramp(स्थिर भग्न निचला, स्थिर भग्न ऊपरी, स्थिर पूर्णांक i0) {
    स्थिर भग्न y = (i0 / 2 - निचला) / sycl::max(0.001f, ऊपरी - निचला);
    वापस 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

ढाँचा <bool forward>
अटल व्योम rope_yarn(स्थिर भग्न theta_extrap, स्थिर भग्न freq_scale,
                      स्थिर rope_corr_dims corr_dims, स्थिर int64_t i0,
                      स्थिर भग्न ext_factor, भग्न mscale, भग्न &cos_theta,
                      भग्न &sin_theta) {
    भग्न theta_interp = freq_scale * theta_extrap;
    भग्न theta = theta_interp;
    अगर (ext_factor != 0.0f) {
        भग्न ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    अगर (!forward) {
        sin_theta *= -1.0f;
    }
}

ढाँचा <bool forward, bool has_ff, typename T, typename D>
अटल व्योम rope_norm(स्थिर T *x, D *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
                      स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02,
                      स्थिर पूर्णांक s03, स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3,
                      स्थिर पूर्णांक n_dims, स्थिर पूर्णांक n_offs, स्थिर int32_t *pos,
                      स्थिर भग्न freq_scale, स्थिर भग्न ext_factor,
                      स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
                      स्थिर भग्न theta_scale, स्थिर भग्न *freq_factors,
                      स्थिर int64_t *row_indices, स्थिर पूर्णांक set_rows_stride) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    अगर (i0 >= ne00) {
        वापस;
    }

    स्थिर पूर्णांक row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    स्थिर uint32_t i3 = row_dst / (ne01 * ne02);
    स्थिर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    स्थिर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णांक idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    स्थिर पूर्णांक ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    अगर (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    स्थिर स्वतः &store_coaelsced = [&](भग्न x0, भग्न x1) {
        अगर constexpr (std::is_same_v<भग्न, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } अन्यथा अगर constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    अगर (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        वापस;
    }

    स्थिर पूर्णांक iw = i0 - n_offs; // relative idx

    स्थिर भग्न theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);

    स्थिर भग्न freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    भग्न cos_theta;
    भग्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    स्थिर भग्न x0 = x[ix + 0];
    स्थिर भग्न x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

ढाँचा <bool forward, bool has_ff, typename T, typename D>
अटल व्योम rope_neox(स्थिर T *x, D *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
                      स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02,
                      स्थिर पूर्णांक s03, स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3,
                      स्थिर पूर्णांक n_dims, स्थिर पूर्णांक n_offs, स्थिर int32_t *pos,
                      स्थिर भग्न freq_scale, स्थिर भग्न ext_factor,
                      स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
                      स्थिर भग्न theta_scale, स्थिर भग्न *freq_factors,
                      स्थिर int64_t *row_indices, स्थिर पूर्णांक set_rows_stride) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    अगर (i0 >= ne00) {
        वापस;
    }

    स्थिर पूर्णांक row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    स्थिर uint32_t i3 = row_dst / (ne01 * ne02);
    स्थिर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    स्थिर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णांक idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    स्थिर पूर्णांक ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    अगर (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    अगर (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        वापस;
    }

    स्थिर पूर्णांक iw = i0 - n_offs; // relative idx

    स्थिर भग्न theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);

    स्थिर भग्न freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    भग्न cos_theta;
    भग्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    स्थिर भग्न x0 = x[ix + n_offs / 2 + 0];
    स्थिर भग्न x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

ढाँचा <bool forward, bool has_ff, typename T>
अटल व्योम rope_multi(स्थिर T *x, T *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
                       स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02,
                       स्थिर पूर्णांक s03, स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3,
                       स्थिर पूर्णांक n_dims, स्थिर पूर्णांक n_offs, स्थिर int32_t *pos,
                       स्थिर भग्न freq_scale, स्थिर भग्न ext_factor,
                       स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
                       स्थिर भग्न theta_scale, स्थिर भग्न *freq_factors,
                       स्थिर mrope_sections sections, स्थिर bool is_imrope) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    अगर (i0 >= ne00) {
        वापस;
    }

    स्थिर पूर्णांक row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    स्थिर uint32_t i3 = row_dst / (ne01 * ne02);
    स्थिर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    स्थिर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णांक idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    स्थिर पूर्णांक ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    अगर (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        वापस;
    }

    स्थिर पूर्णांक iw = i0 - n_offs; // relative idx

    स्थिर पूर्णांक sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    स्थिर पूर्णांक sec_w = sections.v[1] + sections.v[0];
    स्थिर पूर्णांक sector = (iw / 2) % sect_dims;

    भग्न theta_base = 0.0;
    अगर (is_imrope) {
        अगर (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा अगर (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा अगर (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा {
            theta_base = pos[i2 + ne02 * 3] * dpct::घात(theta_scale, iw / 2.0f);
        }
    } अन्यथा {
        अगर (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा अगर (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा अगर (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::घात(theta_scale, iw / 2.0f);
        } अन्यथा अगर (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::घात(theta_scale, iw / 2.0f);
        }
    }

    स्थिर भग्न freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    भग्न cos_theta;
    भग्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    स्थिर भग्न x0 = x[ix + n_offs / 2 + 0];
    स्थिर भग्न x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

ढाँचा <bool forward, bool has_ff, typename T>
अटल व्योम rope_vision(स्थिर T *x, T *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
                        स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02,
                        स्थिर पूर्णांक s03, स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3,
                        स्थिर पूर्णांक n_dims, स्थिर int32_t *pos,
                        स्थिर भग्न freq_scale, स्थिर भग्न ext_factor,
                        स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
                        स्थिर भग्न theta_scale, स्थिर भग्न *freq_factors,
                        स्थिर mrope_sections sections) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    अगर (i0 >= ne00) {
        वापस;
    }

    स्थिर पूर्णांक row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    स्थिर uint32_t i3 = row_dst / (ne01 * ne02);
    स्थिर uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    स्थिर uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    पूर्णांक idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    स्थिर पूर्णांक ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    स्थिर पूर्णांक sect_dims = sections.v[0] + sections.v[1];
    स्थिर पूर्णांक sec_w = sections.v[1] + sections.v[0];
    स्थिर पूर्णांक sector = (i0 / 2) % sect_dims;

    भग्न theta_base = 0.0;
    अगर (sector < sections.v[0]) {
        स्थिर पूर्णांक p = sector;
        theta_base = pos[i2] * dpct::घात(theta_scale, p);
    } अन्यथा अगर (sector >= sections.v[0] && sector < sec_w) {
        स्थिर पूर्णांक p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::घात(theta_scale, p);
    }

    स्थिर भग्न freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    भग्न cos_theta;
    भग्न sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    स्थिर भग्न x0 = x[ix + 0];
    स्थिर भग्न x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

ढाँचा <bool forward, typename T, typename D>
अटल व्योम
rope_norm_sycl(स्थिर T *x, D *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
               स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02, स्थिर पूर्णांक s03,
               स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3, स्थिर पूर्णांक n_dims,
               स्थिर पूर्णांक n_offs, स्थिर पूर्णांक nr, स्थिर int32_t *pos, स्थिर भग्न freq_scale,
               स्थिर भग्न freq_base, स्थिर भग्न ext_factor,
               स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
               स्थिर भग्न *freq_factors, स्थिर int64_t *row_indices,
               स्थिर पूर्णांक set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    स्थिर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_आकार, 1);
    स्थिर पूर्णांक n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_आकार - 1) / (2 * SYCL_ROPE_BLOCK_आकार);
    स्थिर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f / n_dims);

    अगर (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

ढाँचा <bool forward, typename T, typename D>
अटल व्योम
rope_neox_sycl(स्थिर T *x, D *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
               स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02, स्थिर पूर्णांक s03,
               स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3, स्थिर पूर्णांक n_dims,
               स्थिर पूर्णांक n_offs, स्थिर पूर्णांक nr, स्थिर int32_t *pos, स्थिर भग्न freq_scale,
               स्थिर भग्न freq_base, स्थिर भग्न ext_factor,
               स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
               स्थिर भग्न *freq_factors, स्थिर int64_t *row_indices,
               स्थिर पूर्णांक set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    स्थिर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_आकार, 1);
    स्थिर पूर्णांक n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_आकार - 1) / (2 * SYCL_ROPE_BLOCK_आकार);
    स्थिर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f / n_dims);

    अगर (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

ढाँचा <bool forward, typename T>
अटल व्योम
rope_multi_sycl(स्थिर T *x, T *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
                स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02, स्थिर पूर्णांक s03,
                स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3, स्थिर पूर्णांक n_dims,
                स्थिर पूर्णांक n_offs, स्थिर पूर्णांक nr, स्थिर int32_t *pos, स्थिर भग्न freq_scale,
                स्थिर भग्न freq_base, स्थिर भग्न ext_factor,
                स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
                स्थिर भग्न *freq_factors, स्थिर mrope_sections sections,
                स्थिर bool is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    स्थिर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_आकार, 1);
    स्थिर पूर्णांक n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_आकार - 1) / (2 * SYCL_ROPE_BLOCK_आकार);
    स्थिर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f / n_dims);

    अगर (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, false, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, true, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

ढाँचा <bool forward, typename T>
अटल व्योम
rope_vision_sycl(स्थिर T *x, T *dst, स्थिर पूर्णांक ne00, स्थिर पूर्णांक ne01,
                 स्थिर पूर्णांक ne02, स्थिर पूर्णांक s01, स्थिर पूर्णांक s02, स्थिर पूर्णांक s03,
                 स्थिर पूर्णांक s1, स्थिर पूर्णांक s2, स्थिर पूर्णांक s3, स्थिर पूर्णांक n_dims,
                 स्थिर पूर्णांक nr, स्थिर int32_t *pos, स्थिर भग्न freq_scale,
                 स्थिर भग्न freq_base, स्थिर भग्न ext_factor,
                 स्थिर भग्न attn_factor, स्थिर rope_corr_dims corr_dims,
                 स्थिर भग्न *freq_factors, स्थिर mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    स्थिर dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_आकार, 1);
    स्थिर पूर्णांक n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_आकार - 1) / (2 * SYCL_ROPE_BLOCK_आकार);
    स्थिर dpct::dim3 block_nums(nr, n_blocks_x, 1);

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f / n_dims);

    अगर (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, false, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, true, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

ढाँचा <bool forward>
व्योम ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            स्थिर ggml_tensor *set_rows = nullptr) {
    स्थिर ggml_tensor *src0 = dst->src[0];
    स्थिर ggml_tensor *src1 = dst->src[1];
    स्थिर ggml_tensor *src2 = dst->src[2];

    स्थिर भग्न *src0_d = (स्थिर भग्न *)src0->तथ्य;
    स्थिर भग्न *src1_d = (स्थिर भग्न *)src1->तथ्य;

    व्योम *dst_d = dst->तथ्य;
    स्थिर int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    पूर्णांक set_rows_stride = 0;

    अगर (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->तथ्य;
        row_indices = (स्थिर int64_t *)set_rows->src[1]->तथ्य;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_आकार(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    स्थिर int64_t ne00 = src0->ne[0]; // head dims
    स्थिर int64_t ne01 = src0->ne[1]; // num heads
    स्थिर int64_t ne02 = src0->ne[2]; // num heads
    स्थिर int64_t nr = ggml_nrows(src0);

    स्थिर माप_प्रकार s01 = src0->nb[1] / ggml_type_आकार(src0->type);
    स्थिर माप_प्रकार s02 = src0->nb[2] / ggml_type_आकार(src0->type);
    स्थिर माप_प्रकार s03 = src0->nb[3] / ggml_type_आकार(src0->type);

    स्थिर माप_प्रकार s1 = dst->nb[1] / ggml_type_आकार(dst->type);
    स्थिर माप_प्रकार s2 = dst->nb[2] / ggml_type_आकार(dst->type);
    स्थिर माप_प्रकार s3 = dst->nb[3] / ggml_type_आकार(dst->type);

    स्थिर पूर्णांक n_dims = ((int32_t *)dst->op_params)[1];
    स्थिर पूर्णांक mode = ((int32_t *)dst->op_params)[2];
    स्थिर पूर्णांक n_ctx_orig = ((int32_t *)dst->op_params)[4];
    स्थिर पूर्णांक n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    भग्न freq_base;
    भग्न freq_scale;
    भग्न ext_factor;
    भग्न attn_factor;
    भग्न beta_fast;
    भग्न beta_slow;

    स_नकल(&freq_base, (int32_t *)dst->op_params + 5, माप(भग्न));
    स_नकल(&freq_scale, (int32_t *)dst->op_params + 6, माप(भग्न));
    स_नकल(&ext_factor, (int32_t *)dst->op_params + 7, माप(भग्न));
    स_नकल(&attn_factor, (int32_t *)dst->op_params + 8, माप(भग्न));
    स_नकल(&beta_fast, (int32_t *)dst->op_params + 9, माप(भग्न));
    स_नकल(&beta_slow, (int32_t *)dst->op_params + 10, माप(भग्न));
    स_नकल(&sections.v, (int32_t *)dst->op_params + 11, माप(पूर्णांक) * 4);

    स्थिर bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    स्थिर bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    स्थिर bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    स्थिर bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    अगर (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    अगर (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    स्थिर int32_t *pos = (स्थिर int32_t *)src1_d;

    स्थिर भग्न *freq_factors = nullptr;
    अगर (src2 != nullptr) {
        freq_factors = (स्थिर भग्न *)src2->तथ्य;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    अगर (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        अगर (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, भग्न, भग्न>(
                (स्थिर भग्न *)src0_d, (भग्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } अन्यथा अगर (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, भग्न, sycl::half>(
                (स्थिर भग्न *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा अगर (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (स्थिर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा अगर (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        अगर (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((स्थिर भग्न *)src0_d, (भग्न *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } अन्यथा अगर (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (स्थिर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा अगर (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        अगर (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (स्थिर भग्न *)src0_d, (भग्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } अन्यथा अगर (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (स्थिर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } अन्यथा {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        अगर (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, भग्न, भग्न>(
                (स्थिर भग्न *)src0_d, (भग्न *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } अन्यथा अगर (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, भग्न, sycl::half>(
                (स्थिर भग्न *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा अगर (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (स्थिर sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } अन्यथा {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

व्योम ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<true>(ctx, dst);
}

व्योम ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<false>(ctx, dst);
}

व्योम ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<true>(ctx, rope, set_rows);
}
