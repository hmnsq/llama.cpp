#அந்தர்பாவ "rope.hpp"
#அந்தர்பாவ "convert.hpp"
#அந்தர்பாவ "ggml-sycl/common.hpp"
#அந்தர்பாவ "ggml.h"

கட்டமைப்பு rope_corr_dims {
    தசம v[2];
};

கட்டமைப்பு mrope_sections {
    முழு v[4];
};

நிலையான தசம rope_yarn_ramp(மாறிலி தசம கீழ், மாறிலி தசம உயர்வு, மாறிலி முழு i0) {
    மாறிலி தசம y = (i0 / 2 - கீழ்) / sycl::max(0.001f, உயர்வு - கீழ்);
    திரும்பு 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <தருக்க forward>
நிலையான வெறுமை rope_yarn(மாறிலி தசம theta_extrap, மாறிலி தசம freq_scale,
                      மாறிலி rope_corr_dims corr_dims, மாறிலி int64_t i0,
                      மாறிலி தசம ext_factor, தசம mscale, தசம &cos_theta,
                      தசம &sin_theta) {
    தசம theta_interp = freq_scale * theta_extrap;
    தசம theta = theta_interp;
    எனில் (ext_factor != 0.0f) {
        தசம ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    எனில் (!forward) {
        sin_theta *= -1.0f;
    }
}

template <தருக்க forward, தருக்க has_ff, typename T, typename D>
நிலையான வெறுமை rope_norm(மாறிலி T *x, D *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
                      மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02,
                      மாறிலி முழு s03, மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3,
                      மாறிலி முழு n_dims, மாறிலி முழு n_offs, மாறிலி int32_t *pos,
                      மாறிலி தசம freq_scale, மாறிலி தசம ext_factor,
                      மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
                      மாறிலி தசம theta_scale, மாறிலி தசம *freq_factors,
                      மாறிலி int64_t *row_indices, மாறிலி முழு set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    எனில் (i0 >= ne00) {
        திரும்பு;
    }

    மாறிலி முழு row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    மாறிலி uint32_t i3 = row_dst / (ne01 * ne02);
    மாறிலி uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    மாறிலி uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    முழு idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    மாறிலி முழு ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    எனில் (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    மாறிலி auto &store_coaelsced = [&](தசம x0, தசம x1) {
        எனில் constexpr (std::is_same_v<தசம, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } இல்லையெனில் எனில் constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    எனில் (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        திரும்பு;
    }

    மாறிலி முழு iw = i0 - n_offs; // relative idx

    மாறிலி தசம theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    மாறிலி தசம freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    தசம cos_theta;
    தசம sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    மாறிலி தசம x0 = x[ix + 0];
    மாறிலி தசம x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <தருக்க forward, தருக்க has_ff, typename T, typename D>
நிலையான வெறுமை rope_neox(மாறிலி T *x, D *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
                      மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02,
                      மாறிலி முழு s03, மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3,
                      மாறிலி முழு n_dims, மாறிலி முழு n_offs, மாறிலி int32_t *pos,
                      மாறிலி தசம freq_scale, மாறிலி தசம ext_factor,
                      மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
                      மாறிலி தசம theta_scale, மாறிலி தசம *freq_factors,
                      மாறிலி int64_t *row_indices, மாறிலி முழு set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    எனில் (i0 >= ne00) {
        திரும்பு;
    }

    மாறிலி முழு row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    மாறிலி uint32_t i3 = row_dst / (ne01 * ne02);
    மாறிலி uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    மாறிலி uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    முழு idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    மாறிலி முழு ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    எனில் (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    எனில் (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        திரும்பு;
    }

    மாறிலி முழு iw = i0 - n_offs; // relative idx

    மாறிலி தசம theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    மாறிலி தசம freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    தசம cos_theta;
    தசம sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    மாறிலி தசம x0 = x[ix + n_offs / 2 + 0];
    மாறிலி தசம x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <தருக்க forward, தருக்க has_ff, typename T>
நிலையான வெறுமை rope_multi(மாறிலி T *x, T *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
                       மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02,
                       மாறிலி முழு s03, மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3,
                       மாறிலி முழு n_dims, மாறிலி முழு n_offs, மாறிலி int32_t *pos,
                       மாறிலி தசம freq_scale, மாறிலி தசம ext_factor,
                       மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
                       மாறிலி தசம theta_scale, மாறிலி தசம *freq_factors,
                       மாறிலி mrope_sections sections, மாறிலி தருக்க is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    எனில் (i0 >= ne00) {
        திரும்பு;
    }

    மாறிலி முழு row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    மாறிலி uint32_t i3 = row_dst / (ne01 * ne02);
    மாறிலி uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    மாறிலி uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    முழு idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    மாறிலி முழு ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    எனில் (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        திரும்பு;
    }

    மாறிலி முழு iw = i0 - n_offs; // relative idx

    மாறிலி முழு sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    மாறிலி முழு sec_w = sections.v[1] + sections.v[0];
    மாறிலி முழு sector = (iw / 2) % sect_dims;

    தசம theta_base = 0.0;
    எனில் (is_imrope) {
        எனில் (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } இல்லையெனில் எனில் (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } இல்லையெனில் எனில் (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } இல்லையெனில் {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } இல்லையெனில் {
        எனில் (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } இல்லையெனில் எனில் (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } இல்லையெனில் எனில் (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } இல்லையெனில் எனில் (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    மாறிலி தசம freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    தசம cos_theta;
    தசம sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    மாறிலி தசம x0 = x[ix + n_offs / 2 + 0];
    மாறிலி தசம x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <தருக்க forward, தருக்க has_ff, typename T>
நிலையான வெறுமை rope_vision(மாறிலி T *x, T *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
                        மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02,
                        மாறிலி முழு s03, மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3,
                        மாறிலி முழு n_dims, மாறிலி int32_t *pos,
                        மாறிலி தசம freq_scale, மாறிலி தசம ext_factor,
                        மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
                        மாறிலி தசம theta_scale, மாறிலி தசம *freq_factors,
                        மாறிலி mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    எனில் (i0 >= ne00) {
        திரும்பு;
    }

    மாறிலி முழு row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    மாறிலி uint32_t i3 = row_dst / (ne01 * ne02);
    மாறிலி uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    மாறிலி uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    முழு idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    மாறிலி முழு ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    மாறிலி முழு sect_dims = sections.v[0] + sections.v[1];
    மாறிலி முழு sec_w = sections.v[1] + sections.v[0];
    மாறிலி முழு sector = (i0 / 2) % sect_dims;

    தசம theta_base = 0.0;
    எனில் (sector < sections.v[0]) {
        மாறிலி முழு p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } இல்லையெனில் எனில் (sector >= sections.v[0] && sector < sec_w) {
        மாறிலி முழு p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    மாறிலி தசம freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    தசம cos_theta;
    தசம sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    மாறிலி தசம x0 = x[ix + 0];
    மாறிலி தசம x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <தருக்க forward, typename T, typename D>
நிலையான வெறுமை
rope_norm_sycl(மாறிலி T *x, D *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
               மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02, மாறிலி முழு s03,
               மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3, மாறிலி முழு n_dims,
               மாறிலி முழு n_offs, மாறிலி முழு nr, மாறிலி int32_t *pos, மாறிலி தசம freq_scale,
               மாறிலி தசம freq_base, மாறிலி தசம ext_factor,
               மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
               மாறிலி தசம *freq_factors, மாறிலி int64_t *row_indices,
               மாறிலி முழு set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    மாறிலி dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    மாறிலி முழு n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    மாறிலி dpct::dim3 block_nums(nr, n_blocks_x, 1);

    மாறிலி தசம theta_scale = powf(freq_base, -2.0f / n_dims);

    எனில் (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, பொய்>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } இல்லையெனில் {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, உண்மை>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <தருக்க forward, typename T, typename D>
நிலையான வெறுமை
rope_neox_sycl(மாறிலி T *x, D *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
               மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02, மாறிலி முழு s03,
               மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3, மாறிலி முழு n_dims,
               மாறிலி முழு n_offs, மாறிலி முழு nr, மாறிலி int32_t *pos, மாறிலி தசம freq_scale,
               மாறிலி தசம freq_base, மாறிலி தசம ext_factor,
               மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
               மாறிலி தசம *freq_factors, மாறிலி int64_t *row_indices,
               மாறிலி முழு set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    மாறிலி dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    மாறிலி முழு n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    மாறிலி dpct::dim3 block_nums(nr, n_blocks_x, 1);

    மாறிலி தசம theta_scale = powf(freq_base, -2.0f / n_dims);

    எனில் (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, பொய்>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } இல்லையெனில் {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, உண்மை>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <தருக்க forward, typename T>
நிலையான வெறுமை
rope_multi_sycl(மாறிலி T *x, T *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
                மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02, மாறிலி முழு s03,
                மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3, மாறிலி முழு n_dims,
                மாறிலி முழு n_offs, மாறிலி முழு nr, மாறிலி int32_t *pos, மாறிலி தசம freq_scale,
                மாறிலி தசம freq_base, மாறிலி தசம ext_factor,
                மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
                மாறிலி தசம *freq_factors, மாறிலி mrope_sections sections,
                மாறிலி தருக்க is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    மாறிலி dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    மாறிலி முழு n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    மாறிலி dpct::dim3 block_nums(nr, n_blocks_x, 1);

    மாறிலி தசம theta_scale = powf(freq_base, -2.0f / n_dims);

    எனில் (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, பொய், T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } இல்லையெனில் {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, உண்மை, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <தருக்க forward, typename T>
நிலையான வெறுமை
rope_vision_sycl(மாறிலி T *x, T *dst, மாறிலி முழு ne00, மாறிலி முழு ne01,
                 மாறிலி முழு ne02, மாறிலி முழு s01, மாறிலி முழு s02, மாறிலி முழு s03,
                 மாறிலி முழு s1, மாறிலி முழு s2, மாறிலி முழு s3, மாறிலி முழு n_dims,
                 மாறிலி முழு nr, மாறிலி int32_t *pos, மாறிலி தசம freq_scale,
                 மாறிலி தசம freq_base, மாறிலி தசம ext_factor,
                 மாறிலி தசம attn_factor, மாறிலி rope_corr_dims corr_dims,
                 மாறிலி தசம *freq_factors, மாறிலி mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    மாறிலி dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    மாறிலி முழு n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    மாறிலி dpct::dim3 block_nums(nr, n_blocks_x, 1);

    மாறிலி தசம theta_scale = powf(freq_base, -2.0f / n_dims);

    எனில் (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, பொய், T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } இல்லையெனில் {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, உண்மை, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <தருக்க forward>
வெறுமை ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            மாறிலி ggml_tensor *set_rows = nullptr) {
    மாறிலி ggml_tensor *src0 = dst->src[0];
    மாறிலி ggml_tensor *src1 = dst->src[1];
    மாறிலி ggml_tensor *src2 = dst->src[2];

    மாறிலி தசம *src0_d = (மாறிலி தசம *)src0->data;
    மாறிலி தசம *src1_d = (மாறிலி தசம *)src1->data;

    வெறுமை *dst_d = dst->data;
    மாறிலி int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    முழு set_rows_stride = 0;

    எனில் (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (மாறிலி int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    மாறிலி int64_t ne00 = src0->ne[0]; // head dims
    மாறிலி int64_t ne01 = src0->ne[1]; // num heads
    மாறிலி int64_t ne02 = src0->ne[2]; // num heads
    மாறிலி int64_t nr = ggml_nrows(src0);

    மாறிலி size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    மாறிலி size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    மாறிலி size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    மாறிலி size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    மாறிலி size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    மாறிலி size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    மாறிலி முழு n_dims = ((int32_t *)dst->op_params)[1];
    மாறிலி முழு mode = ((int32_t *)dst->op_params)[2];
    மாறிலி முழு n_ctx_orig = ((int32_t *)dst->op_params)[4];
    மாறிலி முழு n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    தசம freq_base;
    தசம freq_scale;
    தசம ext_factor;
    தசம attn_factor;
    தசம beta_fast;
    தசம beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, அளவு(தசம));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, அளவு(தசம));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, அளவு(தசம));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, அளவு(தசம));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, அளவு(தசம));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, அளவு(தசம));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, அளவு(முழு) * 4);

    மாறிலி தருக்க is_neox = mode & GGML_ROPE_TYPE_NEOX;
    மாறிலி தருக்க is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    மாறிலி தருக்க is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    மாறிலி தருக்க is_vision = mode == GGML_ROPE_TYPE_VISION;

    எனில் (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    எனில் (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    மாறிலி int32_t *pos = (மாறிலி int32_t *)src1_d;

    மாறிலி தசம *freq_factors = nullptr;
    எனில் (src2 != nullptr) {
        freq_factors = (மாறிலி தசம *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    எனில் (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        எனில் (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, தசம, தசம>(
                (மாறிலி தசம *)src0_d, (தசம *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, தசம, sycl::half>(
                (மாறிலி தசம *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (மாறிலி sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } இல்லையெனில் {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } இல்லையெனில் எனில் (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        எனில் (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((மாறிலி தசம *)src0_d, (தசம *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (மாறிலி sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } இல்லையெனில் {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } இல்லையெனில் எனில் (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        எனில் (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (மாறிலி தசம *)src0_d, (தசம *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (மாறிலி sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } இல்லையெனில் {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } இல்லையெனில் {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        எனில் (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, தசம, தசம>(
                (மாறிலி தசம *)src0_d, (தசம *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, தசம, sycl::half>(
                (மாறிலி தசம *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (மாறிலி sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } இல்லையெனில் {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

வெறுமை ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<உண்மை>(ctx, dst);
}

வெறுமை ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<பொய்>(ctx, dst);
}

வெறுமை ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<உண்மை>(ctx, rope, set_rows);
}
