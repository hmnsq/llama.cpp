#અન્તર્ભાવ "rope.hpp"
#અન્તર્ભાવ "convert.hpp"
#અન્તર્ભાવ "ggml-sycl/common.hpp"
#અન્તર્ભાવ "ggml.h"

સંરચના rope_corr_dims {
    દશાંશ v[2];
};

સંરચના mrope_sections {
    પૂર્ણાંક v[4];
};

સ્થિર દશાંશ rope_yarn_ramp(અચલ દશાંશ નીચલું, અચલ દશાંશ ઉપલું, અચલ પૂર્ણાંક i0) {
    અચલ દશાંશ y = (i0 / 2 - નીચલું) / sycl::max(0.001f, ઉપલું - નીચલું);
    પાછા 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <તાર્કિક forward>
સ્થિર શૂન્ય rope_yarn(અચલ દશાંશ theta_extrap, અચલ દશાંશ freq_scale,
                      અચલ rope_corr_dims corr_dims, અચલ int64_t i0,
                      અચલ દશાંશ ext_factor, દશાંશ mscale, દશાંશ &cos_theta,
                      દશાંશ &sin_theta) {
    દશાંશ theta_interp = freq_scale * theta_extrap;
    દશાંશ theta = theta_interp;
    જો (ext_factor != 0.0f) {
        દશાંશ ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    જો (!forward) {
        sin_theta *= -1.0f;
    }
}

template <તાર્કિક forward, તાર્કિક has_ff, typename T, typename D>
સ્થિર શૂન્ય rope_norm(અચલ T *x, D *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
                      અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02,
                      અચલ પૂર્ણાંક s03, અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3,
                      અચલ પૂર્ણાંક n_dims, અચલ પૂર્ણાંક n_offs, અચલ int32_t *pos,
                      અચલ દશાંશ freq_scale, અચલ દશાંશ ext_factor,
                      અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
                      અચલ દશાંશ theta_scale, અચલ દશાંશ *freq_factors,
                      અચલ int64_t *row_indices, અચલ પૂર્ણાંક set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ પૂર્ણાંક i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    જો (i0 >= ne00) {
        પાછા;
    }

    અચલ પૂર્ણાંક row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    અચલ uint32_t i3 = row_dst / (ne01 * ne02);
    અચલ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    અચલ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    પૂર્ણાંક idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    અચલ પૂર્ણાંક ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    જો (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    અચલ auto &store_coaelsced = [&](દશાંશ x0, દશાંશ x1) {
        જો constexpr (std::is_same_v<દશાંશ, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } નહીંતર જો constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    જો (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        પાછા;
    }

    અચલ પૂર્ણાંક iw = i0 - n_offs; // relative idx

    અચલ દશાંશ theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    અચલ દશાંશ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    દશાંશ cos_theta;
    દશાંશ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    અચલ દશાંશ x0 = x[ix + 0];
    અચલ દશાંશ x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <તાર્કિક forward, તાર્કિક has_ff, typename T, typename D>
સ્થિર શૂન્ય rope_neox(અચલ T *x, D *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
                      અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02,
                      અચલ પૂર્ણાંક s03, અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3,
                      અચલ પૂર્ણાંક n_dims, અચલ પૂર્ણાંક n_offs, અચલ int32_t *pos,
                      અચલ દશાંશ freq_scale, અચલ દશાંશ ext_factor,
                      અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
                      અચલ દશાંશ theta_scale, અચલ દશાંશ *freq_factors,
                      અચલ int64_t *row_indices, અચલ પૂર્ણાંક set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ પૂર્ણાંક i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    જો (i0 >= ne00) {
        પાછા;
    }

    અચલ પૂર્ણાંક row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    અચલ uint32_t i3 = row_dst / (ne01 * ne02);
    અચલ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    અચલ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    પૂર્ણાંક idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    અચલ પૂર્ણાંક ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    જો (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    જો (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        પાછા;
    }

    અચલ પૂર્ણાંક iw = i0 - n_offs; // relative idx

    અચલ દશાંશ theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    અચલ દશાંશ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    દશાંશ cos_theta;
    દશાંશ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    અચલ દશાંશ x0 = x[ix + n_offs / 2 + 0];
    અચલ દશાંશ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <તાર્કિક forward, તાર્કિક has_ff, typename T>
સ્થિર શૂન્ય rope_multi(અચલ T *x, T *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
                       અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02,
                       અચલ પૂર્ણાંક s03, અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3,
                       અચલ પૂર્ણાંક n_dims, અચલ પૂર્ણાંક n_offs, અચલ int32_t *pos,
                       અચલ દશાંશ freq_scale, અચલ દશાંશ ext_factor,
                       અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
                       અચલ દશાંશ theta_scale, અચલ દશાંશ *freq_factors,
                       અચલ mrope_sections sections, અચલ તાર્કિક is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ પૂર્ણાંક i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    જો (i0 >= ne00) {
        પાછા;
    }

    અચલ પૂર્ણાંક row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    અચલ uint32_t i3 = row_dst / (ne01 * ne02);
    અચલ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    અચલ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    પૂર્ણાંક idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    અચલ પૂર્ણાંક ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    જો (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        પાછા;
    }

    અચલ પૂર્ણાંક iw = i0 - n_offs; // relative idx

    અચલ પૂર્ણાંક sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    અચલ પૂર્ણાંક sec_w = sections.v[1] + sections.v[0];
    અચલ પૂર્ણાંક sector = (iw / 2) % sect_dims;

    દશાંશ theta_base = 0.0;
    જો (is_imrope) {
        જો (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } નહીંતર જો (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } નહીંતર જો (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } નહીંતર {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } નહીંતર {
        જો (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } નહીંતર જો (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } નહીંતર જો (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } નહીંતર જો (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    અચલ દશાંશ freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    દશાંશ cos_theta;
    દશાંશ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    અચલ દશાંશ x0 = x[ix + n_offs / 2 + 0];
    અચલ દશાંશ x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <તાર્કિક forward, તાર્કિક has_ff, typename T>
સ્થિર શૂન્ય rope_vision(અચલ T *x, T *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
                        અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02,
                        અચલ પૂર્ણાંક s03, અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3,
                        અચલ પૂર્ણાંક n_dims, અચલ int32_t *pos,
                        અચલ દશાંશ freq_scale, અચલ દશાંશ ext_factor,
                        અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
                        અચલ દશાંશ theta_scale, અચલ દશાંશ *freq_factors,
                        અચલ mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ પૂર્ણાંક i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    જો (i0 >= ne00) {
        પાછા;
    }

    અચલ પૂર્ણાંક row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    અચલ uint32_t i3 = row_dst / (ne01 * ne02);
    અચલ uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    અચલ uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    પૂર્ણાંક idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    અચલ પૂર્ણાંક ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    અચલ પૂર્ણાંક sect_dims = sections.v[0] + sections.v[1];
    અચલ પૂર્ણાંક sec_w = sections.v[1] + sections.v[0];
    અચલ પૂર્ણાંક sector = (i0 / 2) % sect_dims;

    દશાંશ theta_base = 0.0;
    જો (sector < sections.v[0]) {
        અચલ પૂર્ણાંક p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } નહીંતર જો (sector >= sections.v[0] && sector < sec_w) {
        અચલ પૂર્ણાંક p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    અચલ દશાંશ freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    દશાંશ cos_theta;
    દશાંશ sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    અચલ દશાંશ x0 = x[ix + 0];
    અચલ દશાંશ x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <તાર્કિક forward, typename T, typename D>
સ્થિર શૂન્ય
rope_norm_sycl(અચલ T *x, D *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
               અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02, અચલ પૂર્ણાંક s03,
               અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3, અચલ પૂર્ણાંક n_dims,
               અચલ પૂર્ણાંક n_offs, અચલ પૂર્ણાંક nr, અચલ int32_t *pos, અચલ દશાંશ freq_scale,
               અચલ દશાંશ freq_base, અચલ દશાંશ ext_factor,
               અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
               અચલ દશાંશ *freq_factors, અચલ int64_t *row_indices,
               અચલ પૂર્ણાંક set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    અચલ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    અચલ પૂર્ણાંક n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    અચલ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    અચલ દશાંશ theta_scale = powf(freq_base, -2.0f / n_dims);

    જો (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, ખોટું>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } નહીંતર {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, સાચું>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <તાર્કિક forward, typename T, typename D>
સ્થિર શૂન્ય
rope_neox_sycl(અચલ T *x, D *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
               અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02, અચલ પૂર્ણાંક s03,
               અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3, અચલ પૂર્ણાંક n_dims,
               અચલ પૂર્ણાંક n_offs, અચલ પૂર્ણાંક nr, અચલ int32_t *pos, અચલ દશાંશ freq_scale,
               અચલ દશાંશ freq_base, અચલ દશાંશ ext_factor,
               અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
               અચલ દશાંશ *freq_factors, અચલ int64_t *row_indices,
               અચલ પૂર્ણાંક set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    અચલ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    અચલ પૂર્ણાંક n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    અચલ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    અચલ દશાંશ theta_scale = powf(freq_base, -2.0f / n_dims);

    જો (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, ખોટું>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } નહીંતર {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, સાચું>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <તાર્કિક forward, typename T>
સ્થિર શૂન્ય
rope_multi_sycl(અચલ T *x, T *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
                અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02, અચલ પૂર્ણાંક s03,
                અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3, અચલ પૂર્ણાંક n_dims,
                અચલ પૂર્ણાંક n_offs, અચલ પૂર્ણાંક nr, અચલ int32_t *pos, અચલ દશાંશ freq_scale,
                અચલ દશાંશ freq_base, અચલ દશાંશ ext_factor,
                અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
                અચલ દશાંશ *freq_factors, અચલ mrope_sections sections,
                અચલ તાર્કિક is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    અચલ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    અચલ પૂર્ણાંક n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    અચલ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    અચલ દશાંશ theta_scale = powf(freq_base, -2.0f / n_dims);

    જો (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, ખોટું, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } નહીંતર {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, સાચું, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <તાર્કિક forward, typename T>
સ્થિર શૂન્ય
rope_vision_sycl(અચલ T *x, T *dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક ne01,
                 અચલ પૂર્ણાંક ne02, અચલ પૂર્ણાંક s01, અચલ પૂર્ણાંક s02, અચલ પૂર્ણાંક s03,
                 અચલ પૂર્ણાંક s1, અચલ પૂર્ણાંક s2, અચલ પૂર્ણાંક s3, અચલ પૂર્ણાંક n_dims,
                 અચલ પૂર્ણાંક nr, અચલ int32_t *pos, અચલ દશાંશ freq_scale,
                 અચલ દશાંશ freq_base, અચલ દશાંશ ext_factor,
                 અચલ દશાંશ attn_factor, અચલ rope_corr_dims corr_dims,
                 અચલ દશાંશ *freq_factors, અચલ mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    અચલ dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    અચલ પૂર્ણાંક n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    અચલ dpct::dim3 block_nums(nr, n_blocks_x, 1);

    અચલ દશાંશ theta_scale = powf(freq_base, -2.0f / n_dims);

    જો (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, ખોટું, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } નહીંતર {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, સાચું, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <તાર્કિક forward>
શૂન્ય ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            અચલ ggml_tensor *set_rows = nullptr) {
    અચલ ggml_tensor *src0 = dst->src[0];
    અચલ ggml_tensor *src1 = dst->src[1];
    અચલ ggml_tensor *src2 = dst->src[2];

    અચલ દશાંશ *src0_d = (અચલ દશાંશ *)src0->data;
    અચલ દશાંશ *src1_d = (અચલ દશાંશ *)src1->data;

    શૂન્ય *dst_d = dst->data;
    અચલ int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    પૂર્ણાંક set_rows_stride = 0;

    જો (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (અચલ int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    અચલ int64_t ne00 = src0->ne[0]; // head dims
    અચલ int64_t ne01 = src0->ne[1]; // num heads
    અચલ int64_t ne02 = src0->ne[2]; // num heads
    અચલ int64_t nr = ggml_nrows(src0);

    અચલ size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    અચલ size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    અચલ size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    અચલ size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    અચલ size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    અચલ size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    અચલ પૂર્ણાંક n_dims = ((int32_t *)dst->op_params)[1];
    અચલ પૂર્ણાંક mode = ((int32_t *)dst->op_params)[2];
    અચલ પૂર્ણાંક n_ctx_orig = ((int32_t *)dst->op_params)[4];
    અચલ પૂર્ણાંક n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    દશાંશ freq_base;
    દશાંશ freq_scale;
    દશાંશ ext_factor;
    દશાંશ attn_factor;
    દશાંશ beta_fast;
    દશાંશ beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, કદ(દશાંશ));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, કદ(દશાંશ));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, કદ(દશાંશ));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, કદ(દશાંશ));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, કદ(દશાંશ));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, કદ(દશાંશ));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, કદ(પૂર્ણાંક) * 4);

    અચલ તાર્કિક is_neox = mode & GGML_ROPE_TYPE_NEOX;
    અચલ તાર્કિક is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    અચલ તાર્કિક is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    અચલ તાર્કિક is_vision = mode == GGML_ROPE_TYPE_VISION;

    જો (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    જો (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    અચલ int32_t *pos = (અચલ int32_t *)src1_d;

    અચલ દશાંશ *freq_factors = nullptr;
    જો (src2 != nullptr) {
        freq_factors = (અચલ દશાંશ *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    જો (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        જો (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, દશાંશ, દશાંશ>(
                (અચલ દશાંશ *)src0_d, (દશાંશ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } નહીંતર જો (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, દશાંશ, sycl::half>(
                (અચલ દશાંશ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } નહીંતર જો (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (અચલ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } નહીંતર {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } નહીંતર જો (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        જો (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((અચલ દશાંશ *)src0_d, (દશાંશ *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } નહીંતર જો (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (અચલ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } નહીંતર {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } નહીંતર જો (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        જો (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (અચલ દશાંશ *)src0_d, (દશાંશ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } નહીંતર જો (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (અચલ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } નહીંતર {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } નહીંતર {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        જો (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, દશાંશ, દશાંશ>(
                (અચલ દશાંશ *)src0_d, (દશાંશ *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } નહીંતર જો (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, દશાંશ, sycl::half>(
                (અચલ દશાંશ *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } નહીંતર જો (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (અચલ sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } નહીંતર {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

શૂન્ય ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<સાચું>(ctx, dst);
}

શૂન્ય ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<ખોટું>(ctx, dst);
}

શૂન્ય ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<સાચું>(ctx, rope, set_rows);
}
