#ضمن "rope.hpp"
#ضمن "convert.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml.h"

بنية rope_corr_dims {
    عائم v[2];
};

بنية mrope_sections {
    صحيح v[4];
};

ساكن عائم rope_yarn_ramp(ثابت عائم کم, ثابت عائم بالا, ثابت صحيح i0) {
    ثابت عائم y = (i0 / 2 - کم) / sycl::max(0.001f, بالا - کم);
    أرجع 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

قالب <منطقي forward>
ساكن فراغ rope_yarn(ثابت عائم theta_extrap, ثابت عائم freq_scale,
                      ثابت rope_corr_dims corr_dims, ثابت int64_t i0,
                      ثابت عائم ext_ضریب, عائم mscale, عائم &cos_theta,
                      عائم &sin_theta) {
    عائم theta_interp = freq_scale * theta_extrap;
    عائم theta = theta_interp;
    إذا (ext_ضریب != 0.0f) {
        عائم ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_ضریب;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::لگاریتم(1.0f / freq_scale);
    }
    cos_theta = sycl::کسینوس(theta) * mscale;
    sin_theta = sycl::سینوس(theta) * mscale;
    إذا (!forward) {
        sin_theta *= -1.0f;
    }
}

قالب <منطقي forward, منطقي has_ff, نام_نوع T, نام_نوع D>
ساكن فراغ rope_norm(ثابت T *x, D *dst, ثابت صحيح ne00, ثابت صحيح ne01,
                      ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02,
                      ثابت صحيح s03, ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3,
                      ثابت صحيح n_dims, ثابت صحيح n_offs, ثابت int32_t *pos,
                      ثابت عائم freq_scale, ثابت عائم ext_ضریب,
                      ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
                      ثابت عائم theta_scale, ثابت عائم *freq_factors,
                      ثابت int64_t *ردیف_indices, ثابت صحيح مجموعه_ردیفها_stride) {
    خودکار قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    ثابت صحيح i0 = 2 * (قلم_ct1.get_local_range(1) * قلم_ct1.get_گروه(1) +
                        قلم_ct1.get_local_id(1));

    إذا (i0 >= ne00) {
        أرجع;
    }

    ثابت صحيح ردیف_dst = قلم_ct1.get_local_range(2) * قلم_ct1.get_گروه(2) +
                        قلم_ct1.get_local_id(2);

    ثابت uint32_t i3 = ردیف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (ردیف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = ردیف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحيح idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحيح ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    إذا (مجموعه_ردیفها_stride != 0) {
        idst = i1 * s1 + i0;
        idst += ردیف_indices[i2] * مجموعه_ردیفها_stride;
    }

    ثابت خودکار &store_coaelsced = [&](عائم x0, عائم x1) {
        إذا constexpr (std::is_same_v<عائم, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } وإلا إذا constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    إذا (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        أرجع;
    }

    ثابت صحيح iw = i0 - n_offs; // relative idx

    ثابت عائم theta_پایه = pos[i2] * dpct::توان(theta_scale, iw / 2.0f);

    ثابت عائم freq_ضریب = has_ff ? freq_factors[iw / 2] : 1.0f;

    عائم cos_theta;
    عائم sin_theta;

    rope_yarn<forward>(theta_پایه / freq_ضریب, freq_scale, corr_dims, iw,
                       ext_ضریب, attn_ضریب, cos_theta, sin_theta);

    ثابت عائم x0 = x[ix + 0];
    ثابت عائم x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

قالب <منطقي forward, منطقي has_ff, نام_نوع T, نام_نوع D>
ساكن فراغ rope_neox(ثابت T *x, D *dst, ثابت صحيح ne00, ثابت صحيح ne01,
                      ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02,
                      ثابت صحيح s03, ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3,
                      ثابت صحيح n_dims, ثابت صحيح n_offs, ثابت int32_t *pos,
                      ثابت عائم freq_scale, ثابت عائم ext_ضریب,
                      ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
                      ثابت عائم theta_scale, ثابت عائم *freq_factors,
                      ثابت int64_t *ردیف_indices, ثابت صحيح مجموعه_ردیفها_stride) {
    خودکار قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    ثابت صحيح i0 = 2 * (قلم_ct1.get_local_range(1) * قلم_ct1.get_گروه(1) +
                        قلم_ct1.get_local_id(1));

    إذا (i0 >= ne00) {
        أرجع;
    }

    ثابت صحيح ردیف_dst = قلم_ct1.get_local_range(2) * قلم_ct1.get_گروه(2) +
                        قلم_ct1.get_local_id(2);

    ثابت uint32_t i3 = ردیف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (ردیف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = ردیف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحيح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحيح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    إذا (مجموعه_ردیفها_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += ردیف_indices[i2] * مجموعه_ردیفها_stride;
    }

    إذا (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        أرجع;
    }

    ثابت صحيح iw = i0 - n_offs; // relative idx

    ثابت عائم theta_پایه = pos[i2] * dpct::توان(theta_scale, iw / 2.0f);

    ثابت عائم freq_ضریب = has_ff ? freq_factors[iw / 2] : 1.0f;

    عائم cos_theta;
    عائم sin_theta;

    rope_yarn<forward>(theta_پایه / freq_ضریب, freq_scale, corr_dims, iw,
                       ext_ضریب, attn_ضریب, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ثابت عائم x0 = x[ix + n_offs / 2 + 0];
    ثابت عائم x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

قالب <منطقي forward, منطقي has_ff, نام_نوع T>
ساكن فراغ rope_multi(ثابت T *x, T *dst, ثابت صحيح ne00, ثابت صحيح ne01,
                       ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02,
                       ثابت صحيح s03, ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3,
                       ثابت صحيح n_dims, ثابت صحيح n_offs, ثابت int32_t *pos,
                       ثابت عائم freq_scale, ثابت عائم ext_ضریب,
                       ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
                       ثابت عائم theta_scale, ثابت عائم *freq_factors,
                       ثابت mrope_sections sections, ثابت منطقي is_imrope) {
    خودکار قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    ثابت صحيح i0 = 2 * (قلم_ct1.get_local_range(1) * قلم_ct1.get_گروه(1) +
                        قلم_ct1.get_local_id(1));

    إذا (i0 >= ne00) {
        أرجع;
    }

    ثابت صحيح ردیف_dst = قلم_ct1.get_local_range(2) * قلم_ct1.get_گروه(2) +
                        قلم_ct1.get_local_id(2);

    ثابت uint32_t i3 = ردیف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (ردیف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = ردیف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحيح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحيح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    إذا (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        أرجع;
    }

    ثابت صحيح iw = i0 - n_offs; // relative idx

    ثابت صحيح sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    ثابت صحيح sec_w = sections.v[1] + sections.v[0];
    ثابت صحيح sector = (iw / 2) % sect_dims;

    عائم theta_پایه = 0.0;
    إذا (is_imrope) {
        إذا (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_پایه = pos[i2 + ne02 * 1] * dpct::توان(theta_scale, iw / 2.0f);
        } وإلا إذا (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_پایه = pos[i2 + ne02 * 2] * dpct::توان(theta_scale, iw / 2.0f);
        } وإلا إذا (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_پایه = pos[i2] * dpct::توان(theta_scale, iw / 2.0f);
        } وإلا {
            theta_پایه = pos[i2 + ne02 * 3] * dpct::توان(theta_scale, iw / 2.0f);
        }
    } وإلا {
        إذا (sector < sections.v[0]) {
            theta_پایه = pos[i2] * dpct::توان(theta_scale, iw / 2.0f);
        } وإلا إذا (sector >= sections.v[0] && sector < sec_w) {
            theta_پایه = pos[i2 + ne02 * 1] * dpct::توان(theta_scale, iw / 2.0f);
        } وإلا إذا (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_پایه = pos[i2 + ne02 * 2] * dpct::توان(theta_scale, iw / 2.0f);
        } وإلا إذا (sector >= sec_w + sections.v[2]) {
            theta_پایه = pos[i2 + ne02 * 3] * dpct::توان(theta_scale, iw / 2.0f);
        }
    }

    ثابت عائم freq_ضریب = has_ff ? freq_factors[iw / 2] : 1.0f;

    عائم cos_theta;
    عائم sin_theta;

    rope_yarn<forward>(theta_پایه / freq_ضریب, freq_scale, corr_dims, iw,
                       ext_ضریب, attn_ضریب, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    ثابت عائم x0 = x[ix + n_offs / 2 + 0];
    ثابت عائم x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

قالب <منطقي forward, منطقي has_ff, نام_نوع T>
ساكن فراغ rope_vision(ثابت T *x, T *dst, ثابت صحيح ne00, ثابت صحيح ne01,
                        ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02,
                        ثابت صحيح s03, ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3,
                        ثابت صحيح n_dims, ثابت int32_t *pos,
                        ثابت عائم freq_scale, ثابت عائم ext_ضریب,
                        ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
                        ثابت عائم theta_scale, ثابت عائم *freq_factors,
                        ثابت mrope_sections sections) {
    خودکار قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    ثابت صحيح i0 = 2 * (قلم_ct1.get_local_range(1) * قلم_ct1.get_گروه(1) +
                        قلم_ct1.get_local_id(1));

    إذا (i0 >= ne00) {
        أرجع;
    }

    ثابت صحيح ردیف_dst = قلم_ct1.get_local_range(2) * قلم_ct1.get_گروه(2) +
                        قلم_ct1.get_local_id(2);

    ثابت uint32_t i3 = ردیف_dst / (ne01 * ne02);
    ثابت uint32_t i2 = (ردیف_dst - i3 * ne01 * ne02) / ne01;
    ثابت uint32_t i1 = ردیف_dst - i3 * ne01 * ne02 - i2 * ne01;

    صحيح idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    ثابت صحيح ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ثابت صحيح sect_dims = sections.v[0] + sections.v[1];
    ثابت صحيح sec_w = sections.v[1] + sections.v[0];
    ثابت صحيح sector = (i0 / 2) % sect_dims;

    عائم theta_پایه = 0.0;
    إذا (sector < sections.v[0]) {
        ثابت صحيح p = sector;
        theta_پایه = pos[i2] * dpct::توان(theta_scale, p);
    } وإلا إذا (sector >= sections.v[0] && sector < sec_w) {
        ثابت صحيح p = sector - sections.v[0];
        theta_پایه = pos[i2 + ne02] * dpct::توان(theta_scale, p);
    }

    ثابت عائم freq_ضریب = has_ff ? freq_factors[i0 / 2] : 1.0f;

    عائم cos_theta;
    عائم sin_theta;

    rope_yarn<forward>(theta_پایه / freq_ضریب, freq_scale, corr_dims, i0,
                       ext_ضریب, attn_ضریب, cos_theta, sin_theta);

    ثابت عائم x0 = x[ix + 0];
    ثابت عائم x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

قالب <منطقي forward, نام_نوع T, نام_نوع D>
ساكن فراغ
rope_norm_sycl(ثابت T *x, D *dst, ثابت صحيح ne00, ثابت صحيح ne01,
               ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02, ثابت صحيح s03,
               ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3, ثابت صحيح n_dims,
               ثابت صحيح n_offs, ثابت صحيح nr, ثابت int32_t *pos, ثابت عائم freq_scale,
               ثابت عائم freq_پایه, ثابت عائم ext_ضریب,
               ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
               ثابت عائم *freq_factors, ثابت int64_t *ردیف_indices,
               ثابت صحيح مجموعه_ردیفها_stride, dpct::صف_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلوک_dims(1, SYCL_ROPE_بلوک_اندازه, 1);
    ثابت صحيح n_بلوکها_x =
        (ne00 + 2 * SYCL_ROPE_بلوک_اندازه - 1) / (2 * SYCL_ROPE_بلوک_اندازه);
    ثابت dpct::dim3 بلوک_nums(nr, n_بلوکها_x, 1);

    ثابت عائم theta_scale = powf(freq_پایه, -2.0f / n_dims);

    إذا (freq_factors == اشاره_تهی) {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_norm<forward, خطأ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, ردیف_indices, مجموعه_ردیفها_stride);
            });
    } وإلا {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_norm<forward, درست>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, ردیف_indices, مجموعه_ردیفها_stride);
            });
    }
}

قالب <منطقي forward, نام_نوع T, نام_نوع D>
ساكن فراغ
rope_neox_sycl(ثابت T *x, D *dst, ثابت صحيح ne00, ثابت صحيح ne01,
               ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02, ثابت صحيح s03,
               ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3, ثابت صحيح n_dims,
               ثابت صحيح n_offs, ثابت صحيح nr, ثابت int32_t *pos, ثابت عائم freq_scale,
               ثابت عائم freq_پایه, ثابت عائم ext_ضریب,
               ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
               ثابت عائم *freq_factors, ثابت int64_t *ردیف_indices,
               ثابت صحيح مجموعه_ردیفها_stride, dpct::صف_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلوک_dims(1, SYCL_ROPE_بلوک_اندازه, 1);
    ثابت صحيح n_بلوکها_x =
        (ne00 + 2 * SYCL_ROPE_بلوک_اندازه - 1) / (2 * SYCL_ROPE_بلوک_اندازه);
    ثابت dpct::dim3 بلوک_nums(nr, n_بلوکها_x, 1);

    ثابت عائم theta_scale = powf(freq_پایه, -2.0f / n_dims);

    إذا (freq_factors == اشاره_تهی) {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_neox<forward, خطأ>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, ردیف_indices, مجموعه_ردیفها_stride);
            });
    } وإلا {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_neox<forward, درست>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, ردیف_indices, مجموعه_ردیفها_stride);
            });
    }
}

قالب <منطقي forward, نام_نوع T>
ساكن فراغ
rope_multi_sycl(ثابت T *x, T *dst, ثابت صحيح ne00, ثابت صحيح ne01,
                ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02, ثابت صحيح s03,
                ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3, ثابت صحيح n_dims,
                ثابت صحيح n_offs, ثابت صحيح nr, ثابت int32_t *pos, ثابت عائم freq_scale,
                ثابت عائم freq_پایه, ثابت عائم ext_ضریب,
                ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
                ثابت عائم *freq_factors, ثابت mrope_sections sections,
                ثابت منطقي is_imrope, dpct::صف_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلوک_dims(1, SYCL_ROPE_بلوک_اندازه, 1);
    ثابت صحيح n_بلوکها_x =
        (ne00 + 2 * SYCL_ROPE_بلوک_اندازه - 1) / (2 * SYCL_ROPE_بلوک_اندازه);
    ثابت dpct::dim3 بلوک_nums(nr, n_بلوکها_x, 1);

    ثابت عائم theta_scale = powf(freq_پایه, -2.0f / n_dims);

    إذا (freq_factors == اشاره_تهی) {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_multi<forward, خطأ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } وإلا {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_multi<forward, درست, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

قالب <منطقي forward, نام_نوع T>
ساكن فراغ
rope_vision_sycl(ثابت T *x, T *dst, ثابت صحيح ne00, ثابت صحيح ne01,
                 ثابت صحيح ne02, ثابت صحيح s01, ثابت صحيح s02, ثابت صحيح s03,
                 ثابت صحيح s1, ثابت صحيح s2, ثابت صحيح s3, ثابت صحيح n_dims,
                 ثابت صحيح nr, ثابت int32_t *pos, ثابت عائم freq_scale,
                 ثابت عائم freq_پایه, ثابت عائم ext_ضریب,
                 ثابت عائم attn_ضریب, ثابت rope_corr_dims corr_dims,
                 ثابت عائم *freq_factors, ثابت mrope_sections sections,
                 dpct::صف_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    ثابت dpct::dim3 بلوک_dims(1, SYCL_ROPE_بلوک_اندازه, 1);
    ثابت صحيح n_بلوکها_x =
        (ne00 + 2 * SYCL_ROPE_بلوک_اندازه - 1) / (2 * SYCL_ROPE_بلوک_اندازه);
    ثابت dpct::dim3 بلوک_nums(nr, n_بلوکها_x, 1);

    ثابت عائم theta_scale = powf(freq_پایه, -2.0f / n_dims);

    إذا (freq_factors == اشاره_تهی) {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_vision<forward, خطأ, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } وإلا {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) {
                GGML_استفادهنشده(قلم_ct1);
                rope_vision<forward, درست, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_ضریب, attn_ضریب, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

قالب <منطقي forward>
فراغ ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            ثابت ggml_tensor *مجموعه_ردیفها = اشاره_تهی) {
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت ggml_tensor *src2 = dst->src[2];

    ثابت عائم *src0_d = (ثابت عائم *)src0->داده;
    ثابت عائم *src1_d = (ثابت عائم *)src1->داده;

    فراغ *dst_d = dst->داده;
    ثابت int64_t *ردیف_indices = اشاره_تهی;
    ggml_type dst_type = dst->type;
    صحيح مجموعه_ردیفها_stride = 0;

    إذا (مجموعه_ردیفها != اشاره_تهی) {
        GGML_ASSERT(forward);
        dst_d = مجموعه_ردیفها->داده;
        ردیف_indices = (ثابت int64_t *)مجموعه_ردیفها->src[1]->داده;
        dst_type = مجموعه_ردیفها->type;
        مجموعه_ردیفها_stride = مجموعه_ردیفها->nb[1] / ggml_type_اندازه(مجموعه_ردیفها->type);
    }
    dpct::صف_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    ثابت int64_t ne00 = src0->ne[0]; // head dims
    ثابت int64_t ne01 = src0->ne[1]; // num heads
    ثابت int64_t ne02 = src0->ne[2]; // num heads
    ثابت int64_t nr = ggml_nrows(src0);

    ثابت size_t s01 = src0->nb[1] / ggml_type_اندازه(src0->type);
    ثابت size_t s02 = src0->nb[2] / ggml_type_اندازه(src0->type);
    ثابت size_t s03 = src0->nb[3] / ggml_type_اندازه(src0->type);

    ثابت size_t s1 = dst->nb[1] / ggml_type_اندازه(dst->type);
    ثابت size_t s2 = dst->nb[2] / ggml_type_اندازه(dst->type);
    ثابت size_t s3 = dst->nb[3] / ggml_type_اندازه(dst->type);

    ثابت صحيح n_dims = ((int32_t *)dst->op_params)[1];
    ثابت صحيح mode = ((int32_t *)dst->op_params)[2];
    ثابت صحيح n_ctx_orig = ((int32_t *)dst->op_params)[4];
    ثابت صحيح n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    عائم freq_پایه;
    عائم freq_scale;
    عائم ext_ضریب;
    عائم attn_ضریب;
    عائم beta_fast;
    عائم beta_slow;

    کپی_حافظه(&freq_پایه, (int32_t *)dst->op_params + 5, حجم(عائم));
    کپی_حافظه(&freq_scale, (int32_t *)dst->op_params + 6, حجم(عائم));
    کپی_حافظه(&ext_ضریب, (int32_t *)dst->op_params + 7, حجم(عائم));
    کپی_حافظه(&attn_ضریب, (int32_t *)dst->op_params + 8, حجم(عائم));
    کپی_حافظه(&beta_fast, (int32_t *)dst->op_params + 9, حجم(عائم));
    کپی_حافظه(&beta_slow, (int32_t *)dst->op_params + 10, حجم(عائم));
    کپی_حافظه(&sections.v, (int32_t *)dst->op_params + 11, حجم(صحيح) * 4);

    ثابت منطقي is_neox = mode & GGML_ROPE_TYPE_NEOX;
    ثابت منطقي is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    ثابت منطقي is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ثابت منطقي is_vision = mode == GGML_ROPE_TYPE_VISION;

    إذا (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    إذا (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    ثابت int32_t *pos = (ثابت int32_t *)src1_d;

    ثابت عائم *freq_factors = اشاره_تهی;
    إذا (src2 != اشاره_تهی) {
        freq_factors = (ثابت عائم *)src2->داده;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_پایه, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    إذا (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        إذا (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, عائم, عائم>(
                (ثابت عائم *)src0_d, (عائم *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_پایه,
                ext_ضریب, attn_ضریب, corr_dims, freq_factors, ردیف_indices,
                مجموعه_ردیفها_stride, stream);
        } وإلا إذا (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, عائم, sycl::half>(
                (ثابت عائم *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_پایه, ext_ضریب, attn_ضریب, corr_dims, freq_factors,
                ردیف_indices, مجموعه_ردیفها_stride, stream);
        } وإلا إذا (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_پایه, ext_ضریب, attn_ضریب, corr_dims, freq_factors,
                ردیف_indices, مجموعه_ردیفها_stride, stream);
        } وإلا {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } وإلا إذا (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        إذا (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((ثابت عائم *)src0_d, (عائم *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_پایه,
                                     ext_ضریب, attn_ضریب, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } وإلا إذا (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_پایه, ext_ضریب, attn_ضریب, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } وإلا {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } وإلا إذا (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        إذا (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (ثابت عائم *)src0_d, (عائم *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_پایه,
                ext_ضریب, attn_ضریب, corr_dims, freq_factors, sections,
                stream);
        } وإلا إذا (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_پایه, ext_ضریب, attn_ضریب, corr_dims, freq_factors,
                sections, stream);
        } وإلا {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } وإلا {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        إذا (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, عائم, عائم>(
                (ثابت عائم *)src0_d, (عائم *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_پایه,
                ext_ضریب, attn_ضریب, corr_dims, freq_factors, ردیف_indices,
                مجموعه_ردیفها_stride, stream);
        } وإلا إذا (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, عائم, sycl::half>(
                (ثابت عائم *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_پایه, ext_ضریب, attn_ضریب, corr_dims, freq_factors,
                ردیف_indices, مجموعه_ردیفها_stride, stream);
        } وإلا إذا (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (ثابت sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_پایه, ext_ضریب, attn_ضریب, corr_dims, freq_factors,
                ردیف_indices, مجموعه_ردیفها_stride, stream);
        } وإلا {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

فراغ ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<درست>(ctx, dst);
}

فراغ ggml_sycl_rope_پشت(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<خطأ>(ctx, dst);
}

فراغ ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *مجموعه_ردیفها) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<درست>(ctx, rope, مجموعه_ردیفها);
}
