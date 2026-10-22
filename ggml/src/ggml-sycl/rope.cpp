#অন্তর্ভাৱ "rope.hpp"
#অন্তর্ভাৱ "convert.hpp"
#অন্তর্ভাৱ "ggml-sycl/common.hpp"
#অন্তর্ভাৱ "ggml.h"

সংরচনা rope_corr_dims {
    ভিন্ন v[2];
};

সংরচনা mrope_sections {
    পূর্ণাঙ্ক v[4];
};

স্থির ভিন্ন rope_yarn_ramp(ধ্রুৱ ভিন্ন নিচেৰ, ধ্রুৱ ভিন্ন উপৰেৰ, ধ্রুৱ পূর্ণাঙ্ক i0) {
    ধ্রুৱ ভিন্ন y = (i0 / 2 - নিচেৰ) / sycl::max(0.001f, উপৰেৰ - নিচেৰ);
    ঘূৰাই 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <তার্কিক forward>
স্থির শূন্য rope_yarn(ধ্রুৱ ভিন্ন theta_extrap, ধ্রুৱ ভিন্ন freq_scale,
                      ধ্রুৱ rope_corr_dims corr_dims, ধ্রুৱ int64_t i0,
                      ধ্রুৱ ভিন্ন ext_factor, ভিন্ন mscale, ভিন্ন &cos_theta,
                      ভিন্ন &sin_theta) {
    ভিন্ন theta_interp = freq_scale * theta_extrap;
    ভিন্ন theta = theta_interp;
    যদি (ext_factor != 0.0f) {
        ভিন্ন ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    যদি (!forward) {
        sin_theta *= -1.0f;
    }
}

template <তার্কিক forward, তার্কিক has_ff, typename T, typename D>
স্থির শূন্য rope_norm(ধ্রুৱ T *x, D *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
                      ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02,
                      ধ্রুৱ পূর্ণাঙ্ক s03, ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3,
                      ধ্রুৱ পূর্ণাঙ্ক n_dims, ধ্রুৱ পূর্ণাঙ্ক n_offs, ধ্রুৱ int32_t *pos,
                      ধ্রুৱ ভিন্ন freq_scale, ধ্রুৱ ভিন্ন ext_factor,
                      ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
                      ধ্রুৱ ভিন্ন theta_scale, ধ্রুৱ ভিন্ন *freq_factors,
                      ধ্রুৱ int64_t *row_indices, ধ্রুৱ পূর্ণাঙ্ক set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ পূর্ণাঙ্ক i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুৱ uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুৱ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুৱ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণাঙ্ক idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুৱ পূর্ণাঙ্ক ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    যদি (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    ধ্রুৱ auto &store_coaelsced = [&](ভিন্ন x0, ভিন্ন x1) {
        যদি constexpr (std::is_same_v<ভিন্ন, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } অন্যথা যদি constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    যদি (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক iw = i0 - n_offs; // relative idx

    ধ্রুৱ ভিন্ন theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ধ্রুৱ ভিন্ন freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ভিন্ন cos_theta;
    ভিন্ন sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ধ্রুৱ ভিন্ন x0 = x[ix + 0];
    ধ্রুৱ ভিন্ন x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <তার্কিক forward, তার্কিক has_ff, typename T, typename D>
স্থির শূন্য rope_neox(ধ্রুৱ T *x, D *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
                      ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02,
                      ধ্রুৱ পূর্ণাঙ্ক s03, ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3,
                      ধ্রুৱ পূর্ণাঙ্ক n_dims, ধ্রুৱ পূর্ণাঙ্ক n_offs, ধ্রুৱ int32_t *pos,
                      ধ্রুৱ ভিন্ন freq_scale, ধ্রুৱ ভিন্ন ext_factor,
                      ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
                      ধ্রুৱ ভিন্ন theta_scale, ধ্রুৱ ভিন্ন *freq_factors,
                      ধ্রুৱ int64_t *row_indices, ধ্রুৱ পূর্ণাঙ্ক set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ পূর্ণাঙ্ক i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুৱ uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুৱ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুৱ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণাঙ্ক idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুৱ পূর্ণাঙ্ক ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    যদি (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    যদি (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক iw = i0 - n_offs; // relative idx

    ধ্রুৱ ভিন্ন theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ধ্রুৱ ভিন্ন freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ভিন্ন cos_theta;
    ভিন্ন sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ধ্রুৱ ভিন্ন x0 = x[ix + n_offs / 2 + 0];
    ধ্রুৱ ভিন্ন x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <তার্কিক forward, তার্কিক has_ff, typename T>
স্থির শূন্য rope_multi(ধ্রুৱ T *x, T *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
                       ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02,
                       ধ্রুৱ পূর্ণাঙ্ক s03, ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3,
                       ধ্রুৱ পূর্ণাঙ্ক n_dims, ধ্রুৱ পূর্ণাঙ্ক n_offs, ধ্রুৱ int32_t *pos,
                       ধ্রুৱ ভিন্ন freq_scale, ধ্রুৱ ভিন্ন ext_factor,
                       ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
                       ধ্রুৱ ভিন্ন theta_scale, ধ্রুৱ ভিন্ন *freq_factors,
                       ধ্রুৱ mrope_sections sections, ধ্রুৱ তার্কিক is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ পূর্ণাঙ্ক i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুৱ uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুৱ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুৱ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণাঙ্ক idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুৱ পূর্ণাঙ্ক ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    যদি (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক iw = i0 - n_offs; // relative idx

    ধ্রুৱ পূর্ণাঙ্ক sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ধ্রুৱ পূর্ণাঙ্ক sec_w = sections.v[1] + sections.v[0];
    ধ্রুৱ পূর্ণাঙ্ক sector = (iw / 2) % sect_dims;

    ভিন্ন theta_base = 0.0;
    যদি (is_imrope) {
        যদি (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } অন্যথা যদি (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } অন্যথা {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } অন্যথা {
        যদি (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } অন্যথা যদি (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } অন্যথা যদি (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } অন্যথা যদি (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    ধ্রুৱ ভিন্ন freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    ভিন্ন cos_theta;
    ভিন্ন sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ধ্রুৱ ভিন্ন x0 = x[ix + n_offs / 2 + 0];
    ধ্রুৱ ভিন্ন x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <তার্কিক forward, তার্কিক has_ff, typename T>
স্থির শূন্য rope_vision(ধ্রুৱ T *x, T *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
                        ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02,
                        ধ্রুৱ পূর্ণাঙ্ক s03, ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3,
                        ধ্রুৱ পূর্ণাঙ্ক n_dims, ধ্রুৱ int32_t *pos,
                        ধ্রুৱ ভিন্ন freq_scale, ধ্রুৱ ভিন্ন ext_factor,
                        ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
                        ধ্রুৱ ভিন্ন theta_scale, ধ্রুৱ ভিন্ন *freq_factors,
                        ধ্রুৱ mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ পূর্ণাঙ্ক i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুৱ uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুৱ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুৱ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণাঙ্ক idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুৱ পূর্ণাঙ্ক ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ধ্রুৱ পূর্ণাঙ্ক sect_dims = sections.v[0] + sections.v[1];
    ধ্রুৱ পূর্ণাঙ্ক sec_w = sections.v[1] + sections.v[0];
    ধ্রুৱ পূর্ণাঙ্ক sector = (i0 / 2) % sect_dims;

    ভিন্ন theta_base = 0.0;
    যদি (sector < sections.v[0]) {
        ধ্রুৱ পূর্ণাঙ্ক p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } অন্যথা যদি (sector >= sections.v[0] && sector < sec_w) {
        ধ্রুৱ পূর্ণাঙ্ক p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    ধ্রুৱ ভিন্ন freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    ভিন্ন cos_theta;
    ভিন্ন sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ধ্রুৱ ভিন্ন x0 = x[ix + 0];
    ধ্রুৱ ভিন্ন x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <তার্কিক forward, typename T, typename D>
স্থির শূন্য
rope_norm_sycl(ধ্রুৱ T *x, D *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
               ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02, ধ্রুৱ পূর্ণাঙ্ক s03,
               ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3, ধ্রুৱ পূর্ণাঙ্ক n_dims,
               ধ্রুৱ পূর্ণাঙ্ক n_offs, ধ্রুৱ পূর্ণাঙ্ক nr, ধ্রুৱ int32_t *pos, ধ্রুৱ ভিন্ন freq_scale,
               ধ্রুৱ ভিন্ন freq_base, ধ্রুৱ ভিন্ন ext_factor,
               ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
               ধ্রুৱ ভিন্ন *freq_factors, ধ্রুৱ int64_t *row_indices,
               ধ্রুৱ পূর্ণাঙ্ক set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুৱ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ধ্রুৱ পূর্ণাঙ্ক n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ধ্রুৱ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুৱ ভিন্ন theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, মিছা>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, সঁচা>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <তার্কিক forward, typename T, typename D>
স্থির শূন্য
rope_neox_sycl(ধ্রুৱ T *x, D *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
               ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02, ধ্রুৱ পূর্ণাঙ্ক s03,
               ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3, ধ্রুৱ পূর্ণাঙ্ক n_dims,
               ধ্রুৱ পূর্ণাঙ্ক n_offs, ধ্রুৱ পূর্ণাঙ্ক nr, ধ্রুৱ int32_t *pos, ধ্রুৱ ভিন্ন freq_scale,
               ধ্রুৱ ভিন্ন freq_base, ধ্রুৱ ভিন্ন ext_factor,
               ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
               ধ্রুৱ ভিন্ন *freq_factors, ধ্রুৱ int64_t *row_indices,
               ধ্রুৱ পূর্ণাঙ্ক set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুৱ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ধ্রুৱ পূর্ণাঙ্ক n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ধ্রুৱ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুৱ ভিন্ন theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, মিছা>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, সঁচা>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <তার্কিক forward, typename T>
স্থির শূন্য
rope_multi_sycl(ধ্রুৱ T *x, T *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
                ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02, ধ্রুৱ পূর্ণাঙ্ক s03,
                ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3, ধ্রুৱ পূর্ণাঙ্ক n_dims,
                ধ্রুৱ পূর্ণাঙ্ক n_offs, ধ্রুৱ পূর্ণাঙ্ক nr, ধ্রুৱ int32_t *pos, ধ্রুৱ ভিন্ন freq_scale,
                ধ্রুৱ ভিন্ন freq_base, ধ্রুৱ ভিন্ন ext_factor,
                ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
                ধ্রুৱ ভিন্ন *freq_factors, ধ্রুৱ mrope_sections sections,
                ধ্রুৱ তার্কিক is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুৱ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ধ্রুৱ পূর্ণাঙ্ক n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ধ্রুৱ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুৱ ভিন্ন theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, মিছা, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, সঁচা, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <তার্কিক forward, typename T>
স্থির শূন্য
rope_vision_sycl(ধ্রুৱ T *x, T *dst, ধ্রুৱ পূর্ণাঙ্ক ne00, ধ্রুৱ পূর্ণাঙ্ক ne01,
                 ধ্রুৱ পূর্ণাঙ্ক ne02, ধ্রুৱ পূর্ণাঙ্ক s01, ধ্রুৱ পূর্ণাঙ্ক s02, ধ্রুৱ পূর্ণাঙ্ক s03,
                 ধ্রুৱ পূর্ণাঙ্ক s1, ধ্রুৱ পূর্ণাঙ্ক s2, ধ্রুৱ পূর্ণাঙ্ক s3, ধ্রুৱ পূর্ণাঙ্ক n_dims,
                 ধ্রুৱ পূর্ণাঙ্ক nr, ধ্রুৱ int32_t *pos, ধ্রুৱ ভিন্ন freq_scale,
                 ধ্রুৱ ভিন্ন freq_base, ধ্রুৱ ভিন্ন ext_factor,
                 ধ্রুৱ ভিন্ন attn_factor, ধ্রুৱ rope_corr_dims corr_dims,
                 ধ্রুৱ ভিন্ন *freq_factors, ধ্রুৱ mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুৱ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    ধ্রুৱ পূর্ণাঙ্ক n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    ধ্রুৱ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুৱ ভিন্ন theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, মিছা, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, সঁচা, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <তার্কিক forward>
শূন্য ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ধ্রুৱ ggml_tensor *set_rows = nullptr) {
    ধ্রুৱ ggml_tensor *src0 = dst->src[0];
    ধ্রুৱ ggml_tensor *src1 = dst->src[1];
    ধ্রুৱ ggml_tensor *src2 = dst->src[2];

    ধ্রুৱ ভিন্ন *src0_d = (ধ্রুৱ ভিন্ন *)src0->data;
    ধ্রুৱ ভিন্ন *src1_d = (ধ্রুৱ ভিন্ন *)src1->data;

    শূন্য *dst_d = dst->data;
    ধ্রুৱ int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    পূর্ণাঙ্ক set_rows_stride = 0;

    যদি (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (ধ্রুৱ int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ধ্রুৱ int64_t ne00 = src0->ne[0]; // head dims
    ধ্রুৱ int64_t ne01 = src0->ne[1]; // num heads
    ধ্রুৱ int64_t ne02 = src0->ne[2]; // num heads
    ধ্রুৱ int64_t nr = ggml_nrows(src0);

    ধ্রুৱ size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    ধ্রুৱ size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    ধ্রুৱ size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    ধ্রুৱ size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    ধ্রুৱ size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    ধ্রুৱ size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    ধ্রুৱ পূর্ণাঙ্ক n_dims = ((int32_t *)dst->op_params)[1];
    ধ্রুৱ পূর্ণাঙ্ক mode = ((int32_t *)dst->op_params)[2];
    ধ্রুৱ পূর্ণাঙ্ক n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ধ্রুৱ পূর্ণাঙ্ক n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    ভিন্ন freq_base;
    ভিন্ন freq_scale;
    ভিন্ন ext_factor;
    ভিন্ন attn_factor;
    ভিন্ন beta_fast;
    ভিন্ন beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, পরিমাণ(ভিন্ন));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, পরিমাণ(ভিন্ন));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, পরিমাণ(ভিন্ন));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, পরিমাণ(ভিন্ন));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, পরিমাণ(ভিন্ন));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, পরিমাণ(ভিন্ন));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, পরিমাণ(পূর্ণাঙ্ক) * 4);

    ধ্রুৱ তার্কিক is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ধ্রুৱ তার্কিক is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ধ্রুৱ তার্কিক is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ধ্রুৱ তার্কিক is_vision = mode == GGML_ROPE_TYPE_VISION;

    যদি (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    যদি (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ধ্রুৱ int32_t *pos = (ধ্রুৱ int32_t *)src1_d;

    ধ্রুৱ ভিন্ন *freq_factors = nullptr;
    যদি (src2 != nullptr) {
        freq_factors = (ধ্রুৱ ভিন্ন *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    যদি (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, ভিন্ন, ভিন্ন>(
                (ধ্রুৱ ভিন্ন *)src0_d, (ভিন্ন *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, ভিন্ন, sycl::half>(
                (ধ্রুৱ ভিন্ন *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ধ্রুৱ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } অন্যথা যদি (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ধ্রুৱ ভিন্ন *)src0_d, (ভিন্ন *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ধ্রুৱ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } অন্যথা {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } অন্যথা যদি (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ধ্রুৱ ভিন্ন *)src0_d, (ভিন্ন *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ধ্রুৱ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } অন্যথা {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } অন্যথা {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, ভিন্ন, ভিন্ন>(
                (ধ্রুৱ ভিন্ন *)src0_d, (ভিন্ন *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, ভিন্ন, sycl::half>(
                (ধ্রুৱ ভিন্ন *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ধ্রুৱ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

শূন্য ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<সঁচা>(ctx, dst);
}

শূন্য ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<মিছা>(ctx, dst);
}

শূন্য ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<সঁচা>(ctx, rope, set_rows);
}
