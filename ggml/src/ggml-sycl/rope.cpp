#شامل "rope.hpp"
#شامل "convert.hpp"
#شامل "ggml-sycl/common.hpp"
#شامل "ggml.h"

ساختار rope_corr_dims {
    اعشاری v[2];
};

ساختار mrope_sections {
    صحیح v[4];
};

ایستا اعشاری rope_yarn_ramp(ثابت اعشاری کم, ثابت اعشاری بالا, ثابت صحیح i0) {
    ثابت اعشاری y = (i0 / 2 - کم) / sycl::max(0.001f, بالا - کم);
    بازگشت 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <منطقی forward>
ایستا تهی rope_yarn(ثابت اعشاری theta_extrap, ثابت اعشاری freq_scale,
                      ثابت rope_corr_dims corr_dims, ثابت int64_t i0,
                      ثابت اعشاری ext_factor, اعشاری mscale, اعشاری &cos_theta,
                      اعشاری &sin_theta) {
    اعشاری theta_interp = freq_scale * theta_extrap;
    اعشاری theta = theta_interp;
    اگر (ext_factor != 0.0f) {
        اعشاری ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    اگر (!forward) {
        sin_theta *= -1.0f;
    }
}

template <منطقی forward, منطقی has_ff, typename T, typename D>
ایستا تهی rope_norm(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                      ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                      ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                      ثابت صحیح n_dims, ثابت صحیح n_offs, ثابت int32_t *pos,
                      ثابت اعشاری freq_scale, ثابت اعشاری ext_factor,
                      ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
                      ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                      ثابت int64_t *row_indices, ثابت صحیح set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحیح i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        بازگشت;
    }

    ثابت صحیح row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ثابت uint32_t i3 = row_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    اگر (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    ثابت auto &store_coaelsced = [&](اعشاری x0, اعشاری x1) {
        اگر constexpr (std::is_same_v<اعشاری, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } وگرنه اگر constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    اگر (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        بازگشت;
    }

    ثابت صحیح iw = i0 - n_offs; // relative idx

    ثابت اعشاری theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ثابت اعشاری freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ثابت اعشاری x0 = x[ix + 0];
    ثابت اعشاری x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <منطقی forward, منطقی has_ff, typename T, typename D>
ایستا تهی rope_neox(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                      ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                      ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                      ثابت صحیح n_dims, ثابت صحیح n_offs, ثابت int32_t *pos,
                      ثابت اعشاری freq_scale, ثابت اعشاری ext_factor,
                      ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
                      ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                      ثابت int64_t *row_indices, ثابت صحیح set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحیح i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        بازگشت;
    }

    ثابت صحیح row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ثابت uint32_t i3 = row_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    اگر (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    اگر (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        بازگشت;
    }

    ثابت صحیح iw = i0 - n_offs; // relative idx

    ثابت اعشاری theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ثابت اعشاری freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ثابت اعشاری x0 = x[ix + n_offs / 2 + 0];
    ثابت اعشاری x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <منطقی forward, منطقی has_ff, typename T>
ایستا تهی rope_multi(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                       ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                       ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                       ثابت صحیح n_dims, ثابت صحیح n_offs, ثابت int32_t *pos,
                       ثابت اعشاری freq_scale, ثابت اعشاری ext_factor,
                       ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
                       ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                       ثابت mrope_sections sections, ثابت منطقی is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحیح i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        بازگشت;
    }

    ثابت صحیح row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ثابت uint32_t i3 = row_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    اگر (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        بازگشت;
    }

    ثابت صحیح iw = i0 - n_offs; // relative idx

    ثابت صحیح sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ثابت صحیح sec_w = sections.v[1] + sections.v[0];
    ثابت صحیح sector = (iw / 2) % sect_dims;

    اعشاری theta_base = 0.0;
    اگر (is_imrope) {
        اگر (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } وگرنه اگر (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } وگرنه اگر (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } وگرنه {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } وگرنه {
        اگر (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } وگرنه اگر (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } وگرنه اگر (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } وگرنه اگر (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    ثابت اعشاری freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ثابت اعشاری x0 = x[ix + n_offs / 2 + 0];
    ثابت اعشاری x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <منطقی forward, منطقی has_ff, typename T>
ایستا تهی rope_vision(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                        ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                        ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                        ثابت صحیح n_dims, ثابت int32_t *pos,
                        ثابت اعشاری freq_scale, ثابت اعشاری ext_factor,
                        ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
                        ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                        ثابت mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحیح i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        بازگشت;
    }

    ثابت صحیح row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ثابت uint32_t i3 = row_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ثابت صحیح sect_dims = sections.v[0] + sections.v[1];
    ثابت صحیح sec_w = sections.v[1] + sections.v[0];
    ثابت صحیح sector = (i0 / 2) % sect_dims;

    اعشاری theta_base = 0.0;
    اگر (sector < sections.v[0]) {
        ثابت صحیح p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } وگرنه اگر (sector >= sections.v[0] && sector < sec_w) {
        ثابت صحیح p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    ثابت اعشاری freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ثابت اعشاری x0 = x[ix + 0];
    ثابت اعشاری x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <منطقی forward, typename T, typename D>
ایستا تهی
rope_norm_sycl(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
               ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
               ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
               ثابت صحیح n_offs, ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
               ثابت اعشاری freq_base, ثابت اعشاری ext_factor,
               ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
               ثابت اعشاری *freq_factors, ثابت int64_t *row_indices,
               ثابت صحیح set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ثابت صحیح n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ثابت dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ثابت اعشاری theta_scale = powf(freq_base, -2.0f / n_dims);

    اگر (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, نادرست>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } وگرنه {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, درست>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <منطقی forward, typename T, typename D>
ایستا تهی
rope_neox_sycl(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
               ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
               ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
               ثابت صحیح n_offs, ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
               ثابت اعشاری freq_base, ثابت اعشاری ext_factor,
               ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
               ثابت اعشاری *freq_factors, ثابت int64_t *row_indices,
               ثابت صحیح set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ثابت صحیح n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ثابت dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ثابت اعشاری theta_scale = powf(freq_base, -2.0f / n_dims);

    اگر (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, نادرست>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } وگرنه {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, درست>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <منطقی forward, typename T>
ایستا تهی
rope_multi_sycl(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
                ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
                ثابت صحیح n_offs, ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
                ثابت اعشاری freq_base, ثابت اعشاری ext_factor,
                ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
                ثابت اعشاری *freq_factors, ثابت mrope_sections sections,
                ثابت منطقی is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ثابت صحیح n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ثابت dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ثابت اعشاری theta_scale = powf(freq_base, -2.0f / n_dims);

    اگر (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, نادرست, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } وگرنه {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, درست, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <منطقی forward, typename T>
ایستا تهی
rope_vision_sycl(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                 ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
                 ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
                 ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
                 ثابت اعشاری freq_base, ثابت اعشاری ext_factor,
                 ثابت اعشاری attn_factor, ثابت rope_corr_dims corr_dims,
                 ثابت اعشاری *freq_factors, ثابت mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ثابت صحیح n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ثابت dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ثابت اعشاری theta_scale = powf(freq_base, -2.0f / n_dims);

    اگر (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, نادرست, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } وگرنه {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, درست, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <منطقی forward>
تهی ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ثابت ggml_tensor *set_rows = nullptr) {
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت ggml_tensor *src2 = dst->src[2];

    ثابت اعشاری *src0_d = (ثابت اعشاری *)src0->data;
    ثابت اعشاری *src1_d = (ثابت اعشاری *)src1->data;

    تهی *dst_d = dst->data;
    ثابت int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    صحیح set_rows_stride = 0;

    اگر (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (ثابت int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ثابت int64_t ne00 = src0->ne[0]; // head dims
    ثابت int64_t ne01 = src0->ne[1]; // num heads
    ثابت int64_t ne02 = src0->ne[2]; // num heads
    ثابت int64_t nr = ggml_nrows(src0);

    ثابت size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    ثابت size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    ثابت size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    ثابت size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    ثابت size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    ثابت size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    ثابت صحیح n_dims = ((int32_t *)dst->op_params)[1];
    ثابت صحیح mode = ((int32_t *)dst->op_params)[2];
    ثابت صحیح n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ثابت صحیح n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    اعشاری freq_base;
    اعشاری freq_scale;
    اعشاری ext_factor;
    اعشاری attn_factor;
    اعشاری beta_fast;
    اعشاری beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, اندازه(اعشاری));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, اندازه(اعشاری));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, اندازه(اعشاری));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, اندازه(اعشاری));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, اندازه(اعشاری));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, اندازه(اعشاری));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, اندازه(صحیح) * 4);

    ثابت منطقی is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ثابت منطقی is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ثابت منطقی is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ثابت منطقی is_vision = mode == GGML_ROPE_TYPE_VISION;

    اگر (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    اگر (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ثابت int32_t *pos = (ثابت int32_t *)src1_d;

    ثابت اعشاری *freq_factors = nullptr;
    اگر (src2 != nullptr) {
        freq_factors = (ثابت اعشاری *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    اگر (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, اعشاری, اعشاری>(
                (ثابت اعشاری *)src0_d, (اعشاری *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } وگرنه اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, اعشاری, sycl::half>(
                (ثابت اعشاری *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } وگرنه اگر (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } وگرنه {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } وگرنه اگر (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ثابت اعشاری *)src0_d, (اعشاری *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } وگرنه اگر (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } وگرنه {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } وگرنه اگر (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ثابت اعشاری *)src0_d, (اعشاری *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } وگرنه اگر (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } وگرنه {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } وگرنه {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, اعشاری, اعشاری>(
                (ثابت اعشاری *)src0_d, (اعشاری *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } وگرنه اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, اعشاری, sycl::half>(
                (ثابت اعشاری *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } وگرنه اگر (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } وگرنه {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

تهی ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<درست>(ctx, dst);
}

تهی ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<نادرست>(ctx, dst);
}

تهی ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<درست>(ctx, rope, set_rows);
}
