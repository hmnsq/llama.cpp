#অন্তর্ভাব "rope.hpp"
#অন্তর্ভাব "convert.hpp"
#অন্তর্ভাব "ggml-sycl/common.hpp"
#অন্তর্ভাব "ggml.h"

কাঠামো rope_corr_dims {
    দশমিক v[2];
};

কাঠামো mrope_sections {
    পূর্ণসংখ্যা v[4];
};

স্থির দশমিক rope_yarn_ramp(ধ্রুবক দশমিক নিচের, ধ্রুবক দশমিক উপরের, ধ্রুবক পূর্ণসংখ্যা i0) {
    ধ্রুবক দশমিক y = (i0 / 2 - নিচের) / sycl::max(0.001f, উপরের - নিচের);
    ফেরত 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <যৌক্তিক forward>
স্থির শূন্য rope_yarn(ধ্রুবক দশমিক theta_extrap, ধ্রুবক দশমিক freq_scale,
                      ধ্রুবক rope_corr_dims corr_dims, ধ্রুবক int64_t i0,
                      ধ্রুবক দশমিক ext_factor, দশমিক mscale, দশমিক &cos_theta,
                      দশমিক &sin_theta) {
    দশমিক theta_interp = freq_scale * theta_extrap;
    দশমিক theta = theta_interp;
    যদি (ext_factor != 0.0f) {
        দশমিক ramp_mix =
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

template <যৌক্তিক forward, যৌক্তিক has_ff, typename T, typename D>
স্থির শূন্য rope_norm(ধ্রুবক T *x, D *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                      ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02,
                      ধ্রুবক পূর্ণসংখ্যা s03, ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3,
                      ধ্রুবক পূর্ণসংখ্যা n_dims, ধ্রুবক পূর্ণসংখ্যা n_offs, ধ্রুবক int32_t *pos,
                      ধ্রুবক দশমিক freq_scale, ধ্রুবক দশমিক ext_factor,
                      ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
                      ধ্রুবক দশমিক theta_scale, ধ্রুবক দশমিক *freq_factors,
                      ধ্রুবক int64_t *row_indices, ধ্রুবক পূর্ণসংখ্যা set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক পূর্ণসংখ্যা i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুবক uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুবক uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুবক uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণসংখ্যা idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুবক পূর্ণসংখ্যা ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    যদি (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    ধ্রুবক auto &store_coaelsced = [&](দশমিক x0, দশমিক x1) {
        যদি constexpr (std::is_same_v<দশমিক, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } অন্যথা যদি constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    যদি (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা iw = i0 - n_offs; // relative idx

    ধ্রুবক দশমিক theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ধ্রুবক দশমিক freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    দশমিক cos_theta;
    দশমিক sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ধ্রুবক দশমিক x0 = x[ix + 0];
    ধ্রুবক দশমিক x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <যৌক্তিক forward, যৌক্তিক has_ff, typename T, typename D>
স্থির শূন্য rope_neox(ধ্রুবক T *x, D *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                      ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02,
                      ধ্রুবক পূর্ণসংখ্যা s03, ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3,
                      ধ্রুবক পূর্ণসংখ্যা n_dims, ধ্রুবক পূর্ণসংখ্যা n_offs, ধ্রুবক int32_t *pos,
                      ধ্রুবক দশমিক freq_scale, ধ্রুবক দশমিক ext_factor,
                      ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
                      ধ্রুবক দশমিক theta_scale, ধ্রুবক দশমিক *freq_factors,
                      ধ্রুবক int64_t *row_indices, ধ্রুবক পূর্ণসংখ্যা set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক পূর্ণসংখ্যা i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুবক uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুবক uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুবক uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণসংখ্যা idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুবক পূর্ণসংখ্যা ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    যদি (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    যদি (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা iw = i0 - n_offs; // relative idx

    ধ্রুবক দশমিক theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    ধ্রুবক দশমিক freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    দশমিক cos_theta;
    দশমিক sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ধ্রুবক দশমিক x0 = x[ix + n_offs / 2 + 0];
    ধ্রুবক দশমিক x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <যৌক্তিক forward, যৌক্তিক has_ff, typename T>
স্থির শূন্য rope_multi(ধ্রুবক T *x, T *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                       ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02,
                       ধ্রুবক পূর্ণসংখ্যা s03, ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3,
                       ধ্রুবক পূর্ণসংখ্যা n_dims, ধ্রুবক পূর্ণসংখ্যা n_offs, ধ্রুবক int32_t *pos,
                       ধ্রুবক দশমিক freq_scale, ধ্রুবক দশমিক ext_factor,
                       ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
                       ধ্রুবক দশমিক theta_scale, ধ্রুবক দশমিক *freq_factors,
                       ধ্রুবক mrope_sections sections, ধ্রুবক যৌক্তিক is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক পূর্ণসংখ্যা i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুবক uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুবক uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুবক uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণসংখ্যা idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুবক পূর্ণসংখ্যা ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    যদি (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা iw = i0 - n_offs; // relative idx

    ধ্রুবক পূর্ণসংখ্যা sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ধ্রুবক পূর্ণসংখ্যা sec_w = sections.v[1] + sections.v[0];
    ধ্রুবক পূর্ণসংখ্যা sector = (iw / 2) % sect_dims;

    দশমিক theta_base = 0.0;
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

    ধ্রুবক দশমিক freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    দশমিক cos_theta;
    দশমিক sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ধ্রুবক দশমিক x0 = x[ix + n_offs / 2 + 0];
    ধ্রুবক দশমিক x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <যৌক্তিক forward, যৌক্তিক has_ff, typename T>
স্থির শূন্য rope_vision(ধ্রুবক T *x, T *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                        ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02,
                        ধ্রুবক পূর্ণসংখ্যা s03, ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3,
                        ধ্রুবক পূর্ণসংখ্যা n_dims, ধ্রুবক int32_t *pos,
                        ধ্রুবক দশমিক freq_scale, ধ্রুবক দশমিক ext_factor,
                        ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
                        ধ্রুবক দশমিক theta_scale, ধ্রুবক দশমিক *freq_factors,
                        ধ্রুবক mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক পূর্ণসংখ্যা i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    যদি (i0 >= ne00) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    ধ্রুবক uint32_t i3 = row_dst / (ne01 * ne02);
    ধ্রুবক uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    ধ্রুবক uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    পূর্ণসংখ্যা idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ধ্রুবক পূর্ণসংখ্যা ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ধ্রুবক পূর্ণসংখ্যা sect_dims = sections.v[0] + sections.v[1];
    ধ্রুবক পূর্ণসংখ্যা sec_w = sections.v[1] + sections.v[0];
    ধ্রুবক পূর্ণসংখ্যা sector = (i0 / 2) % sect_dims;

    দশমিক theta_base = 0.0;
    যদি (sector < sections.v[0]) {
        ধ্রুবক পূর্ণসংখ্যা p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } অন্যথা যদি (sector >= sections.v[0] && sector < sec_w) {
        ধ্রুবক পূর্ণসংখ্যা p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    ধ্রুবক দশমিক freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    দশমিক cos_theta;
    দশমিক sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    ধ্রুবক দশমিক x0 = x[ix + 0];
    ধ্রুবক দশমিক x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <যৌক্তিক forward, typename T, typename D>
স্থির শূন্য
rope_norm_sycl(ধ্রুবক T *x, D *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
               ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02, ধ্রুবক পূর্ণসংখ্যা s03,
               ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3, ধ্রুবক পূর্ণসংখ্যা n_dims,
               ধ্রুবক পূর্ণসংখ্যা n_offs, ধ্রুবক পূর্ণসংখ্যা nr, ধ্রুবক int32_t *pos, ধ্রুবক দশমিক freq_scale,
               ধ্রুবক দশমিক freq_base, ধ্রুবক দশমিক ext_factor,
               ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
               ধ্রুবক দশমিক *freq_factors, ধ্রুবক int64_t *row_indices,
               ধ্রুবক পূর্ণসংখ্যা set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুবক dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_আকার, 1);
    ধ্রুবক পূর্ণসংখ্যা n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_আকার - 1) / (2 * SYCL_ROPE_BLOCK_আকার);
    ধ্রুবক dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুবক দশমিক theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, মিথ্যা>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, সত্য>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <যৌক্তিক forward, typename T, typename D>
স্থির শূন্য
rope_neox_sycl(ধ্রুবক T *x, D *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
               ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02, ধ্রুবক পূর্ণসংখ্যা s03,
               ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3, ধ্রুবক পূর্ণসংখ্যা n_dims,
               ধ্রুবক পূর্ণসংখ্যা n_offs, ধ্রুবক পূর্ণসংখ্যা nr, ধ্রুবক int32_t *pos, ধ্রুবক দশমিক freq_scale,
               ধ্রুবক দশমিক freq_base, ধ্রুবক দশমিক ext_factor,
               ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
               ধ্রুবক দশমিক *freq_factors, ধ্রুবক int64_t *row_indices,
               ধ্রুবক পূর্ণসংখ্যা set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুবক dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_আকার, 1);
    ধ্রুবক পূর্ণসংখ্যা n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_আকার - 1) / (2 * SYCL_ROPE_BLOCK_আকার);
    ধ্রুবক dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুবক দশমিক theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, মিথ্যা>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, সত্য>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <যৌক্তিক forward, typename T>
স্থির শূন্য
rope_multi_sycl(ধ্রুবক T *x, T *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02, ধ্রুবক পূর্ণসংখ্যা s03,
                ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3, ধ্রুবক পূর্ণসংখ্যা n_dims,
                ধ্রুবক পূর্ণসংখ্যা n_offs, ধ্রুবক পূর্ণসংখ্যা nr, ধ্রুবক int32_t *pos, ধ্রুবক দশমিক freq_scale,
                ধ্রুবক দশমিক freq_base, ধ্রুবক দশমিক ext_factor,
                ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
                ধ্রুবক দশমিক *freq_factors, ধ্রুবক mrope_sections sections,
                ধ্রুবক যৌক্তিক is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুবক dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_আকার, 1);
    ধ্রুবক পূর্ণসংখ্যা n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_আকার - 1) / (2 * SYCL_ROPE_BLOCK_আকার);
    ধ্রুবক dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুবক দশমিক theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, মিথ্যা, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, সত্য, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <যৌক্তিক forward, typename T>
স্থির শূন্য
rope_vision_sycl(ধ্রুবক T *x, T *dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                 ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা s01, ধ্রুবক পূর্ণসংখ্যা s02, ধ্রুবক পূর্ণসংখ্যা s03,
                 ধ্রুবক পূর্ণসংখ্যা s1, ধ্রুবক পূর্ণসংখ্যা s2, ধ্রুবক পূর্ণসংখ্যা s3, ধ্রুবক পূর্ণসংখ্যা n_dims,
                 ধ্রুবক পূর্ণসংখ্যা nr, ধ্রুবক int32_t *pos, ধ্রুবক দশমিক freq_scale,
                 ধ্রুবক দশমিক freq_base, ধ্রুবক দশমিক ext_factor,
                 ধ্রুবক দশমিক attn_factor, ধ্রুবক rope_corr_dims corr_dims,
                 ধ্রুবক দশমিক *freq_factors, ধ্রুবক mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ধ্রুবক dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_আকার, 1);
    ধ্রুবক পূর্ণসংখ্যা n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_আকার - 1) / (2 * SYCL_ROPE_BLOCK_আকার);
    ধ্রুবক dpct::dim3 block_nums(nr, n_blocks_x, 1);

    ধ্রুবক দশমিক theta_scale = powf(freq_base, -2.0f / n_dims);

    যদি (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, মিথ্যা, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, সত্য, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <যৌক্তিক forward>
শূন্য ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ধ্রুবক ggml_tensor *set_rows = nullptr) {
    ধ্রুবক ggml_tensor *src0 = dst->src[0];
    ধ্রুবক ggml_tensor *src1 = dst->src[1];
    ধ্রুবক ggml_tensor *src2 = dst->src[2];

    ধ্রুবক দশমিক *src0_d = (ধ্রুবক দশমিক *)src0->তথ্য;
    ধ্রুবক দশমিক *src1_d = (ধ্রুবক দশমিক *)src1->তথ্য;

    শূন্য *dst_d = dst->তথ্য;
    ধ্রুবক int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    পূর্ণসংখ্যা set_rows_stride = 0;

    যদি (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->তথ্য;
        row_indices = (ধ্রুবক int64_t *)set_rows->src[1]->তথ্য;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_আকার(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ধ্রুবক int64_t ne00 = src0->ne[0]; // head dims
    ধ্রুবক int64_t ne01 = src0->ne[1]; // num heads
    ধ্রুবক int64_t ne02 = src0->ne[2]; // num heads
    ধ্রুবক int64_t nr = ggml_nrows(src0);

    ধ্রুবক size_t s01 = src0->nb[1] / ggml_type_আকার(src0->type);
    ধ্রুবক size_t s02 = src0->nb[2] / ggml_type_আকার(src0->type);
    ধ্রুবক size_t s03 = src0->nb[3] / ggml_type_আকার(src0->type);

    ধ্রুবক size_t s1 = dst->nb[1] / ggml_type_আকার(dst->type);
    ধ্রুবক size_t s2 = dst->nb[2] / ggml_type_আকার(dst->type);
    ধ্রুবক size_t s3 = dst->nb[3] / ggml_type_আকার(dst->type);

    ধ্রুবক পূর্ণসংখ্যা n_dims = ((int32_t *)dst->op_params)[1];
    ধ্রুবক পূর্ণসংখ্যা mode = ((int32_t *)dst->op_params)[2];
    ধ্রুবক পূর্ণসংখ্যা n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ধ্রুবক পূর্ণসংখ্যা n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    দশমিক freq_base;
    দশমিক freq_scale;
    দশমিক ext_factor;
    দশমিক attn_factor;
    দশমিক beta_fast;
    দশমিক beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, আকার(দশমিক));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, আকার(দশমিক));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, আকার(দশমিক));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, আকার(দশমিক));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, আকার(দশমিক));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, আকার(দশমিক));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, আকার(পূর্ণসংখ্যা) * 4);

    ধ্রুবক যৌক্তিক is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ধ্রুবক যৌক্তিক is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ধ্রুবক যৌক্তিক is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ধ্রুবক যৌক্তিক is_vision = mode == GGML_ROPE_TYPE_VISION;

    যদি (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    যদি (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ধ্রুবক int32_t *pos = (ধ্রুবক int32_t *)src1_d;

    ধ্রুবক দশমিক *freq_factors = nullptr;
    যদি (src2 != nullptr) {
        freq_factors = (ধ্রুবক দশমিক *)src2->তথ্য;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    যদি (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, দশমিক, দশমিক>(
                (ধ্রুবক দশমিক *)src0_d, (দশমিক *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, দশমিক, sycl::half>(
                (ধ্রুবক দশমিক *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ধ্রুবক sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } অন্যথা যদি (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ধ্রুবক দশমিক *)src0_d, (দশমিক *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ধ্রুবক sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
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
                (ধ্রুবক দশমিক *)src0_d, (দশমিক *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ধ্রুবক sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } অন্যথা {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } অন্যথা {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, দশমিক, দশমিক>(
                (ধ্রুবক দশমিক *)src0_d, (দশমিক *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, দশমিক, sycl::half>(
                (ধ্রুবক দশমিক *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ধ্রুবক sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
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

    ggml_sycl_op_rope_impl<সত্য>(ctx, dst);
}

শূন্য ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<মিথ্যা>(ctx, dst);
}

শূন্য ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<সত্য>(ctx, rope, set_rows);
}
