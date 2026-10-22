#شامل "rope.hpp"
#شامل "convert.hpp"
#شامل "ggml-sycl/common.hpp"
#شامل "ggml.h"

ساخت rope_corr_dims {
    اعشاری v[2];
};

ساخت mrope_sections {
    صحیح v[4];
};

ساکن اعشاری rope_yarn_ramp(ثابت اعشاری کم, ثابت اعشاری بلند, ثابت صحیح i0) {
    ثابت اعشاری y = (i0 / 2 - کم) / sycl::max(0.001f, بلند - کم);
    واپس 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

سانچہ <منطقی forward>
ساکن خالی rope_yarn(ثابت اعشاری theta_extrap, ثابت اعشاری freq_scale,
                      ثابت rope_corr_dims corr_dims, ثابت int64_t i0,
                      ثابت اعشاری ext_ضربی_عدد, اعشاری mscale, اعشاری &cos_theta,
                      اعشاری &sin_theta) {
    اعشاری theta_interp = freq_scale * theta_extrap;
    اعشاری theta = theta_interp;
    اگر (ext_ضربی_عدد != 0.0f) {
        اعشاری ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_ضربی_عدد;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::لوگارتھم(1.0f / freq_scale);
    }
    cos_theta = sycl::جیب_تمام(theta) * mscale;
    sin_theta = sycl::جیب(theta) * mscale;
    اگر (!forward) {
        sin_theta *= -1.0f;
    }
}

سانچہ <منطقی forward, منطقی has_ff, قسم_نام T, قسم_نام D>
ساکن خالی rope_norm(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                      ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                      ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                      ثابت صحیح n_dims, ثابت صحیح n_offs, ثابت int32_t *pos,
                      ثابت اعشاری freq_scale, ثابت اعشاری ext_ضربی_عدد,
                      ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
                      ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                      ثابت int64_t *صف_indices, ثابت صحیح سیٹ_صفیں_stride) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح i0 = 2 * (شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                        شے_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        واپس;
    }

    ثابت صحیح صف_dst = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                        شے_ct1.get_local_id(2);

    ثابت uint32_t i3 = صف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (صف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = صف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    اگر (سیٹ_صفیں_stride != 0) {
        idst = i1 * s1 + i0;
        idst += صف_indices[i2] * سیٹ_صفیں_stride;
    }

    ثابت خودکار &store_coaelsced = [&](اعشاری x0, اعشاری x1) {
        اگر constexpr (std::is_same_v<اعشاری, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } ورنہ اگر constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    اگر (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        واپس;
    }

    ثابت صحیح iw = i0 - n_offs; // relative idx

    ثابت اعشاری theta_بنیاد = pos[i2] * dpct::قوت(theta_scale, iw / 2.0f);

    ثابت اعشاری freq_ضربی_عدد = has_ff ? freq_factors[iw / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_بنیاد / freq_ضربی_عدد, freq_scale, corr_dims, iw,
                       ext_ضربی_عدد, attn_ضربی_عدد, cos_theta, sin_theta);

    ثابت اعشاری x0 = x[ix + 0];
    ثابت اعشاری x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

سانچہ <منطقی forward, منطقی has_ff, قسم_نام T, قسم_نام D>
ساکن خالی rope_neox(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                      ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                      ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                      ثابت صحیح n_dims, ثابت صحیح n_offs, ثابت int32_t *pos,
                      ثابت اعشاری freq_scale, ثابت اعشاری ext_ضربی_عدد,
                      ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
                      ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                      ثابت int64_t *صف_indices, ثابت صحیح سیٹ_صفیں_stride) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح i0 = 2 * (شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                        شے_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        واپس;
    }

    ثابت صحیح صف_dst = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                        شے_ct1.get_local_id(2);

    ثابت uint32_t i3 = صف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (صف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = صف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    اگر (سیٹ_صفیں_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += صف_indices[i2] * سیٹ_صفیں_stride;
    }

    اگر (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        واپس;
    }

    ثابت صحیح iw = i0 - n_offs; // relative idx

    ثابت اعشاری theta_بنیاد = pos[i2] * dpct::قوت(theta_scale, iw / 2.0f);

    ثابت اعشاری freq_ضربی_عدد = has_ff ? freq_factors[iw / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_بنیاد / freq_ضربی_عدد, freq_scale, corr_dims, iw,
                       ext_ضربی_عدد, attn_ضربی_عدد, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ثابت اعشاری x0 = x[ix + n_offs / 2 + 0];
    ثابت اعشاری x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

سانچہ <منطقی forward, منطقی has_ff, قسم_نام T>
ساکن خالی rope_multi(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                       ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                       ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                       ثابت صحیح n_dims, ثابت صحیح n_offs, ثابت int32_t *pos,
                       ثابت اعشاری freq_scale, ثابت اعشاری ext_ضربی_عدد,
                       ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
                       ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                       ثابت mrope_sections sections, ثابت منطقی is_imrope) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح i0 = 2 * (شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                        شے_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        واپس;
    }

    ثابت صحیح صف_dst = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                        شے_ct1.get_local_id(2);

    ثابت uint32_t i3 = صف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (صف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = صف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    اگر (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        واپس;
    }

    ثابت صحیح iw = i0 - n_offs; // relative idx

    ثابت صحیح sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ثابت صحیح sec_w = sections.v[1] + sections.v[0];
    ثابت صحیح sector = (iw / 2) % sect_dims;

    اعشاری theta_بنیاد = 0.0;
    اگر (is_imrope) {
        اگر (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_بنیاد = pos[i2 + ne02 * 1] * dpct::قوت(theta_scale, iw / 2.0f);
        } ورنہ اگر (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_بنیاد = pos[i2 + ne02 * 2] * dpct::قوت(theta_scale, iw / 2.0f);
        } ورنہ اگر (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_بنیاد = pos[i2] * dpct::قوت(theta_scale, iw / 2.0f);
        } ورنہ {
            theta_بنیاد = pos[i2 + ne02 * 3] * dpct::قوت(theta_scale, iw / 2.0f);
        }
    } ورنہ {
        اگر (sector < sections.v[0]) {
            theta_بنیاد = pos[i2] * dpct::قوت(theta_scale, iw / 2.0f);
        } ورنہ اگر (sector >= sections.v[0] && sector < sec_w) {
            theta_بنیاد = pos[i2 + ne02 * 1] * dpct::قوت(theta_scale, iw / 2.0f);
        } ورنہ اگر (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_بنیاد = pos[i2 + ne02 * 2] * dpct::قوت(theta_scale, iw / 2.0f);
        } ورنہ اگر (sector >= sec_w + sections.v[2]) {
            theta_بنیاد = pos[i2 + ne02 * 3] * dpct::قوت(theta_scale, iw / 2.0f);
        }
    }

    ثابت اعشاری freq_ضربی_عدد = has_ff ? freq_factors[iw / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_بنیاد / freq_ضربی_عدد, freq_scale, corr_dims, iw,
                       ext_ضربی_عدد, attn_ضربی_عدد, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ثابت اعشاری x0 = x[ix + n_offs / 2 + 0];
    ثابت اعشاری x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

سانچہ <منطقی forward, منطقی has_ff, قسم_نام T>
ساکن خالی rope_vision(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                        ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02,
                        ثابت صحیح s03, ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3,
                        ثابت صحیح n_dims, ثابت int32_t *pos,
                        ثابت اعشاری freq_scale, ثابت اعشاری ext_ضربی_عدد,
                        ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
                        ثابت اعشاری theta_scale, ثابت اعشاری *freq_factors,
                        ثابت mrope_sections sections) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح i0 = 2 * (شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                        شے_ct1.get_local_id(1));

    اگر (i0 >= ne00) {
        واپس;
    }

    ثابت صحیح صف_dst = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                        شے_ct1.get_local_id(2);

    ثابت uint32_t i3 = صف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (صف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = صف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحیح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحیح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ثابت صحیح sect_dims = sections.v[0] + sections.v[1];
    ثابت صحیح sec_w = sections.v[1] + sections.v[0];
    ثابت صحیح sector = (i0 / 2) % sect_dims;

    اعشاری theta_بنیاد = 0.0;
    اگر (sector < sections.v[0]) {
        ثابت صحیح p = sector;
        theta_بنیاد = pos[i2] * dpct::قوت(theta_scale, p);
    } ورنہ اگر (sector >= sections.v[0] && sector < sec_w) {
        ثابت صحیح p = sector - sections.v[0];
        theta_بنیاد = pos[i2 + ne02] * dpct::قوت(theta_scale, p);
    }

    ثابت اعشاری freq_ضربی_عدد = has_ff ? freq_factors[i0 / 2] : 1.0f;

    اعشاری cos_theta;
    اعشاری sin_theta;

    rope_yarn<forward>(theta_بنیاد / freq_ضربی_عدد, freq_scale, corr_dims, i0,
                       ext_ضربی_عدد, attn_ضربی_عدد, cos_theta, sin_theta);

    ثابت اعشاری x0 = x[ix + 0];
    ثابت اعشاری x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

سانچہ <منطقی forward, قسم_نام T, قسم_نام D>
ساکن خالی
rope_norm_sycl(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
               ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
               ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
               ثابت صحیح n_offs, ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
               ثابت اعشاری freq_بنیاد, ثابت اعشاری ext_ضربی_عدد,
               ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
               ثابت اعشاری *freq_factors, ثابت int64_t *صف_indices,
               ثابت صحیح سیٹ_صفیں_stride, dpct::قطار_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلاک_dims(1, SYCL_ROPE_بلاک_سائز, 1);
    ثابت صحیح n_بلاکیں_x =
        (ne00 + 2 * SYCL_ROPE_بلاک_سائز - 1) / (2 * SYCL_ROPE_بلاک_سائز);
    ثابت dpct::dim3 بلاک_nums(nr, n_بلاکیں_x, 1);

    ثابت اعشاری theta_scale = powf(freq_بنیاد, -2.0f / n_dims);

    اگر (freq_factors == خالی_اشاریہ) {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_norm<forward, جھوٹ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, صف_indices, سیٹ_صفیں_stride);
            });
    } ورنہ {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_norm<forward, سچ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, صف_indices, سیٹ_صفیں_stride);
            });
    }
}

سانچہ <منطقی forward, قسم_نام T, قسم_نام D>
ساکن خالی
rope_neox_sycl(ثابت T *x, D *dst, ثابت صحیح ne00, ثابت صحیح ne01,
               ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
               ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
               ثابت صحیح n_offs, ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
               ثابت اعشاری freq_بنیاد, ثابت اعشاری ext_ضربی_عدد,
               ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
               ثابت اعشاری *freq_factors, ثابت int64_t *صف_indices,
               ثابت صحیح سیٹ_صفیں_stride, dpct::قطار_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلاک_dims(1, SYCL_ROPE_بلاک_سائز, 1);
    ثابت صحیح n_بلاکیں_x =
        (ne00 + 2 * SYCL_ROPE_بلاک_سائز - 1) / (2 * SYCL_ROPE_بلاک_سائز);
    ثابت dpct::dim3 بلاک_nums(nr, n_بلاکیں_x, 1);

    ثابت اعشاری theta_scale = powf(freq_بنیاد, -2.0f / n_dims);

    اگر (freq_factors == خالی_اشاریہ) {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_neox<forward, جھوٹ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, صف_indices, سیٹ_صفیں_stride);
            });
    } ورنہ {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_neox<forward, سچ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, صف_indices, سیٹ_صفیں_stride);
            });
    }
}

سانچہ <منطقی forward, قسم_نام T>
ساکن خالی
rope_multi_sycl(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
                ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
                ثابت صحیح n_offs, ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
                ثابت اعشاری freq_بنیاد, ثابت اعشاری ext_ضربی_عدد,
                ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
                ثابت اعشاری *freq_factors, ثابت mrope_sections sections,
                ثابت منطقی is_imrope, dpct::قطار_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلاک_dims(1, SYCL_ROPE_بلاک_سائز, 1);
    ثابت صحیح n_بلاکیں_x =
        (ne00 + 2 * SYCL_ROPE_بلاک_سائز - 1) / (2 * SYCL_ROPE_بلاک_سائز);
    ثابت dpct::dim3 بلاک_nums(nr, n_بلاکیں_x, 1);

    ثابت اعشاری theta_scale = powf(freq_بنیاد, -2.0f / n_dims);

    اگر (freq_factors == خالی_اشاریہ) {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_multi<forward, جھوٹ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } ورنہ {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_multi<forward, سچ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

سانچہ <منطقی forward, قسم_نام T>
ساکن خالی
rope_vision_sycl(ثابت T *x, T *dst, ثابت صحیح ne00, ثابت صحیح ne01,
                 ثابت صحیح ne02, ثابت صحیح s01, ثابت صحیح s02, ثابت صحیح s03,
                 ثابت صحیح s1, ثابت صحیح s2, ثابت صحیح s3, ثابت صحیح n_dims,
                 ثابت صحیح nr, ثابت int32_t *pos, ثابت اعشاری freq_scale,
                 ثابت اعشاری freq_بنیاد, ثابت اعشاری ext_ضربی_عدد,
                 ثابت اعشاری attn_ضربی_عدد, ثابت rope_corr_dims corr_dims,
                 ثابت اعشاری *freq_factors, ثابت mrope_sections sections,
                 dpct::قطار_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلاک_dims(1, SYCL_ROPE_بلاک_سائز, 1);
    ثابت صحیح n_بلاکیں_x =
        (ne00 + 2 * SYCL_ROPE_بلاک_سائز - 1) / (2 * SYCL_ROPE_بلاک_سائز);
    ثابت dpct::dim3 بلاک_nums(nr, n_بلاکیں_x, 1);

    ثابت اعشاری theta_scale = powf(freq_بنیاد, -2.0f / n_dims);

    اگر (freq_factors == خالی_اشاریہ) {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_vision<forward, جھوٹ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } ورنہ {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                GGML_غیر_مستعمل(شے_ct1);
                rope_vision<forward, سچ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

سانچہ <منطقی forward>
خالی ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ثابت ggml_tensor *سیٹ_صفیں = خالی_اشاریہ) {
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت ggml_tensor *src2 = dst->src[2];

    ثابت اعشاری *src0_d = (ثابت اعشاری *)src0->ڈیٹا;
    ثابت اعشاری *src1_d = (ثابت اعشاری *)src1->ڈیٹا;

    خالی *dst_d = dst->ڈیٹا;
    ثابت int64_t *صف_indices = خالی_اشاریہ;
    ggml_type dst_type = dst->type;
    صحیح سیٹ_صفیں_stride = 0;

    اگر (سیٹ_صفیں != خالی_اشاریہ) {
        GGML_ASSERT(forward);
        dst_d = سیٹ_صفیں->ڈیٹا;
        صف_indices = (ثابت int64_t *)سیٹ_صفیں->src[1]->ڈیٹا;
        dst_type = سیٹ_صفیں->type;
        سیٹ_صفیں_stride = سیٹ_صفیں->nb[1] / ggml_type_سائز(سیٹ_صفیں->type);
    }
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ثابت int64_t ne00 = src0->ne[0]; // head dims
    ثابت int64_t ne01 = src0->ne[1]; // num heads
    ثابت int64_t ne02 = src0->ne[2]; // num heads
    ثابت int64_t nr = ggml_nrows(src0);

    ثابت size_t s01 = src0->nb[1] / ggml_type_سائز(src0->type);
    ثابت size_t s02 = src0->nb[2] / ggml_type_سائز(src0->type);
    ثابت size_t s03 = src0->nb[3] / ggml_type_سائز(src0->type);

    ثابت size_t s1 = dst->nb[1] / ggml_type_سائز(dst->type);
    ثابت size_t s2 = dst->nb[2] / ggml_type_سائز(dst->type);
    ثابت size_t s3 = dst->nb[3] / ggml_type_سائز(dst->type);

    ثابت صحیح n_dims = ((int32_t *)dst->op_params)[1];
    ثابت صحیح mode = ((int32_t *)dst->op_params)[2];
    ثابت صحیح n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ثابت صحیح n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    اعشاری freq_بنیاد;
    اعشاری freq_scale;
    اعشاری ext_ضربی_عدد;
    اعشاری attn_ضربی_عدد;
    اعشاری beta_fast;
    اعشاری beta_slow;

    حافظہ_نقل(&freq_بنیاد, (int32_t *)dst->op_params + 5, حجم(اعشاری));
    حافظہ_نقل(&freq_scale, (int32_t *)dst->op_params + 6, حجم(اعشاری));
    حافظہ_نقل(&ext_ضربی_عدد, (int32_t *)dst->op_params + 7, حجم(اعشاری));
    حافظہ_نقل(&attn_ضربی_عدد, (int32_t *)dst->op_params + 8, حجم(اعشاری));
    حافظہ_نقل(&beta_fast, (int32_t *)dst->op_params + 9, حجم(اعشاری));
    حافظہ_نقل(&beta_slow, (int32_t *)dst->op_params + 10, حجم(اعشاری));
    حافظہ_نقل(&sections.v, (int32_t *)dst->op_params + 11, حجم(صحیح) * 4);

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

    ثابت اعشاری *freq_factors = خالی_اشاریہ;
    اگر (src2 != خالی_اشاریہ) {
        freq_factors = (ثابت اعشاری *)src2->ڈیٹا;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنیاد, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    اگر (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, اعشاری, اعشاری>(
                (ثابت اعشاری *)src0_d, (اعشاری *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_بنیاد,
                ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors, صف_indices,
                سیٹ_صفیں_stride, stream);
        } ورنہ اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, اعشاری, sycl::half>(
                (ثابت اعشاری *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_بنیاد, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors,
                صف_indices, سیٹ_صفیں_stride, stream);
        } ورنہ اگر (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_بنیاد, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors,
                صف_indices, سیٹ_صفیں_stride, stream);
        } ورنہ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ورنہ اگر (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ثابت اعشاری *)src0_d, (اعشاری *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_بنیاد,
                                     ext_ضربی_عدد, attn_ضربی_عدد, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } ورنہ اگر (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_بنیاد, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } ورنہ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ورنہ اگر (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ثابت اعشاری *)src0_d, (اعشاری *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_بنیاد,
                ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors, sections,
                stream);
        } ورنہ اگر (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_بنیاد, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors,
                sections, stream);
        } ورنہ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } ورنہ {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, اعشاری, اعشاری>(
                (ثابت اعشاری *)src0_d, (اعشاری *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_بنیاد,
                ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors, صف_indices,
                سیٹ_صفیں_stride, stream);
        } ورنہ اگر (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, اعشاری, sycl::half>(
                (ثابت اعشاری *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_بنیاد, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors,
                صف_indices, سیٹ_صفیں_stride, stream);
        } ورنہ اگر (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_بنیاد, ext_ضربی_عدد, attn_ضربی_عدد, corr_dims, freq_factors,
                صف_indices, سیٹ_صفیں_stride, stream);
        } ورنہ {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

خالی ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<سچ>(ctx, dst);
}

خالی ggml_sycl_rope_عقب(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<جھوٹ>(ctx, dst);
}

خالی ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *سیٹ_صفیں) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<سچ>(ctx, rope, سیٹ_صفیں);
}
