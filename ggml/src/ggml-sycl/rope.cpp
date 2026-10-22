#అన్తర్భావ "rope.hpp"
#అన్తర్భావ "convert.hpp"
#అన్తర్భావ "ggml-sycl/common.hpp"
#అన్తర్భావ "ggml.h"

నిర్మాణం rope_corr_dims {
    దశాంశం v[2];
};

నిర్మాణం mrope_sections {
    పూర్ణాంకం v[4];
};

స్థిర దశాంశం rope_yarn_ramp(స్థిరరాశి దశాంశం దిగువ, స్థిరరాశి దశాంశం ఎగువ, స్థిరరాశి పూర్ణాంకం i0) {
    స్థిరరాశి దశాంశం y = (i0 / 2 - దిగువ) / sycl::max(0.001f, ఎగువ - దిగువ);
    తిరిగి 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

మూస <తార్కిక forward>
స్థిర శూన్య rope_yarn(స్థిరరాశి దశాంశం theta_extrap, స్థిరరాశి దశాంశం freq_scale,
                      స్థిరరాశి rope_corr_dims corr_dims, స్థిరరాశి int64_t i0,
                      స్థిరరాశి దశాంశం ext_గుణకం, దశాంశం mscale, దశాంశం &cos_theta,
                      దశాంశం &sin_theta) {
    దశాంశం theta_interp = freq_scale * theta_extrap;
    దశాంశం theta = theta_interp;
    ఒకవేళ (ext_గుణకం != 0.0f) {
        దశాంశం ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_గుణకం;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::సంవర్గమానం(1.0f / freq_scale);
    }
    cos_theta = sycl::కోజ్యా(theta) * mscale;
    sin_theta = sycl::జ్యా(theta) * mscale;
    ఒకవేళ (!forward) {
        sin_theta *= -1.0f;
    }
}

మూస <తార్కిక forward, తార్కిక has_ff, రకం_పేరు T, రకం_పేరు D>
స్థిర శూన్య rope_norm(స్థిరరాశి T *x, D *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
                      స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02,
                      స్థిరరాశి పూర్ణాంకం s03, స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3,
                      స్థిరరాశి పూర్ణాంకం n_dims, స్థిరరాశి పూర్ణాంకం n_offs, స్థిరరాశి int32_t *pos,
                      స్థిరరాశి దశాంశం freq_scale, స్థిరరాశి దశాంశం ext_గుణకం,
                      స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
                      స్థిరరాశి దశాంశం theta_scale, స్థిరరాశి దశాంశం *freq_factors,
                      స్థిరరాశి int64_t *అడ్డువరుస_indices, స్థిరరాశి పూర్ణాంకం సమితి_అడ్డువరుసలు_stride) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం i0 = 2 * (అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                        అంశం_ct1.get_local_id(1));

    ఒకవేళ (i0 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                        అంశం_ct1.get_local_id(2);

    స్థిరరాశి uint32_t i3 = అడ్డువరుస_dst / (ne01 * ne02);
    స్థిరరాశి uint32_t i2 = (అడ్డువరుస_dst - i3 * ne01 * ne02) / ne01;
    స్థిరరాశి uint32_t i1 = అడ్డువరుస_dst - i3 * ne01 * ne02 - i2 * ne01;

    పూర్ణాంకం idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    స్థిరరాశి పూర్ణాంకం ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    ఒకవేళ (సమితి_అడ్డువరుసలు_stride != 0) {
        idst = i1 * s1 + i0;
        idst += అడ్డువరుస_indices[i2] * సమితి_అడ్డువరుసలు_stride;
    }

    స్థిరరాశి స్వయంచాలక &store_coaelsced = [&](దశాంశం x0, దశాంశం x1) {
        ఒకవేళ constexpr (std::is_same_v<దశాంశం, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } లేకపోతే ఒకవేళ constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    ఒకవేళ (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం iw = i0 - n_offs; // relative idx

    స్థిరరాశి దశాంశం theta_పునాది = pos[i2] * dpct::ఘాతం(theta_scale, iw / 2.0f);

    స్థిరరాశి దశాంశం freq_గుణకం = has_ff ? freq_factors[iw / 2] : 1.0f;

    దశాంశం cos_theta;
    దశాంశం sin_theta;

    rope_yarn<forward>(theta_పునాది / freq_గుణకం, freq_scale, corr_dims, iw,
                       ext_గుణకం, attn_గుణకం, cos_theta, sin_theta);

    స్థిరరాశి దశాంశం x0 = x[ix + 0];
    స్థిరరాశి దశాంశం x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

మూస <తార్కిక forward, తార్కిక has_ff, రకం_పేరు T, రకం_పేరు D>
స్థిర శూన్య rope_neox(స్థిరరాశి T *x, D *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
                      స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02,
                      స్థిరరాశి పూర్ణాంకం s03, స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3,
                      స్థిరరాశి పూర్ణాంకం n_dims, స్థిరరాశి పూర్ణాంకం n_offs, స్థిరరాశి int32_t *pos,
                      స్థిరరాశి దశాంశం freq_scale, స్థిరరాశి దశాంశం ext_గుణకం,
                      స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
                      స్థిరరాశి దశాంశం theta_scale, స్థిరరాశి దశాంశం *freq_factors,
                      స్థిరరాశి int64_t *అడ్డువరుస_indices, స్థిరరాశి పూర్ణాంకం సమితి_అడ్డువరుసలు_stride) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం i0 = 2 * (అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                        అంశం_ct1.get_local_id(1));

    ఒకవేళ (i0 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                        అంశం_ct1.get_local_id(2);

    స్థిరరాశి uint32_t i3 = అడ్డువరుస_dst / (ne01 * ne02);
    స్థిరరాశి uint32_t i2 = (అడ్డువరుస_dst - i3 * ne01 * ne02) / ne01;
    స్థిరరాశి uint32_t i1 = అడ్డువరుస_dst - i3 * ne01 * ne02 - i2 * ne01;

    పూర్ణాంకం idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    స్థిరరాశి పూర్ణాంకం ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ఒకవేళ (సమితి_అడ్డువరుసలు_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += అడ్డువరుస_indices[i2] * సమితి_అడ్డువరుసలు_stride;
    }

    ఒకవేళ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం iw = i0 - n_offs; // relative idx

    స్థిరరాశి దశాంశం theta_పునాది = pos[i2] * dpct::ఘాతం(theta_scale, iw / 2.0f);

    స్థిరరాశి దశాంశం freq_గుణకం = has_ff ? freq_factors[iw / 2] : 1.0f;

    దశాంశం cos_theta;
    దశాంశం sin_theta;

    rope_yarn<forward>(theta_పునాది / freq_గుణకం, freq_scale, corr_dims, iw,
                       ext_గుణకం, attn_గుణకం, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    స్థిరరాశి దశాంశం x0 = x[ix + n_offs / 2 + 0];
    స్థిరరాశి దశాంశం x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

మూస <తార్కిక forward, తార్కిక has_ff, రకం_పేరు T>
స్థిర శూన్య rope_multi(స్థిరరాశి T *x, T *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
                       స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02,
                       స్థిరరాశి పూర్ణాంకం s03, స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3,
                       స్థిరరాశి పూర్ణాంకం n_dims, స్థిరరాశి పూర్ణాంకం n_offs, స్థిరరాశి int32_t *pos,
                       స్థిరరాశి దశాంశం freq_scale, స్థిరరాశి దశాంశం ext_గుణకం,
                       స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
                       స్థిరరాశి దశాంశం theta_scale, స్థిరరాశి దశాంశం *freq_factors,
                       స్థిరరాశి mrope_sections sections, స్థిరరాశి తార్కిక is_imrope) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం i0 = 2 * (అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                        అంశం_ct1.get_local_id(1));

    ఒకవేళ (i0 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                        అంశం_ct1.get_local_id(2);

    స్థిరరాశి uint32_t i3 = అడ్డువరుస_dst / (ne01 * ne02);
    స్థిరరాశి uint32_t i2 = (అడ్డువరుస_dst - i3 * ne01 * ne02) / ne01;
    స్థిరరాశి uint32_t i1 = అడ్డువరుస_dst - i3 * ne01 * ne02 - i2 * ne01;

    పూర్ణాంకం idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    స్థిరరాశి పూర్ణాంకం ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ఒకవేళ (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం iw = i0 - n_offs; // relative idx

    స్థిరరాశి పూర్ణాంకం sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    స్థిరరాశి పూర్ణాంకం sec_w = sections.v[1] + sections.v[0];
    స్థిరరాశి పూర్ణాంకం sector = (iw / 2) % sect_dims;

    దశాంశం theta_పునాది = 0.0;
    ఒకవేళ (is_imrope) {
        ఒకవేళ (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_పునాది = pos[i2 + ne02 * 1] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        } లేకపోతే ఒకవేళ (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_పునాది = pos[i2 + ne02 * 2] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        } లేకపోతే ఒకవేళ (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_పునాది = pos[i2] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        } లేకపోతే {
            theta_పునాది = pos[i2 + ne02 * 3] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        }
    } లేకపోతే {
        ఒకవేళ (sector < sections.v[0]) {
            theta_పునాది = pos[i2] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        } లేకపోతే ఒకవేళ (sector >= sections.v[0] && sector < sec_w) {
            theta_పునాది = pos[i2 + ne02 * 1] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        } లేకపోతే ఒకవేళ (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_పునాది = pos[i2 + ne02 * 2] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        } లేకపోతే ఒకవేళ (sector >= sec_w + sections.v[2]) {
            theta_పునాది = pos[i2 + ne02 * 3] * dpct::ఘాతం(theta_scale, iw / 2.0f);
        }
    }

    స్థిరరాశి దశాంశం freq_గుణకం = has_ff ? freq_factors[iw / 2] : 1.0f;

    దశాంశం cos_theta;
    దశాంశం sin_theta;

    rope_yarn<forward>(theta_పునాది / freq_గుణకం, freq_scale, corr_dims, iw,
                       ext_గుణకం, attn_గుణకం, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    స్థిరరాశి దశాంశం x0 = x[ix + n_offs / 2 + 0];
    స్థిరరాశి దశాంశం x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

మూస <తార్కిక forward, తార్కిక has_ff, రకం_పేరు T>
స్థిర శూన్య rope_vision(స్థిరరాశి T *x, T *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
                        స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02,
                        స్థిరరాశి పూర్ణాంకం s03, స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3,
                        స్థిరరాశి పూర్ణాంకం n_dims, స్థిరరాశి int32_t *pos,
                        స్థిరరాశి దశాంశం freq_scale, స్థిరరాశి దశాంశం ext_గుణకం,
                        స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
                        స్థిరరాశి దశాంశం theta_scale, స్థిరరాశి దశాంశం *freq_factors,
                        స్థిరరాశి mrope_sections sections) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం i0 = 2 * (అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                        అంశం_ct1.get_local_id(1));

    ఒకవేళ (i0 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                        అంశం_ct1.get_local_id(2);

    స్థిరరాశి uint32_t i3 = అడ్డువరుస_dst / (ne01 * ne02);
    స్థిరరాశి uint32_t i2 = (అడ్డువరుస_dst - i3 * ne01 * ne02) / ne01;
    స్థిరరాశి uint32_t i1 = అడ్డువరుస_dst - i3 * ne01 * ne02 - i2 * ne01;

    పూర్ణాంకం idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    స్థిరరాశి పూర్ణాంకం ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    స్థిరరాశి పూర్ణాంకం sect_dims = sections.v[0] + sections.v[1];
    స్థిరరాశి పూర్ణాంకం sec_w = sections.v[1] + sections.v[0];
    స్థిరరాశి పూర్ణాంకం sector = (i0 / 2) % sect_dims;

    దశాంశం theta_పునాది = 0.0;
    ఒకవేళ (sector < sections.v[0]) {
        స్థిరరాశి పూర్ణాంకం p = sector;
        theta_పునాది = pos[i2] * dpct::ఘాతం(theta_scale, p);
    } లేకపోతే ఒకవేళ (sector >= sections.v[0] && sector < sec_w) {
        స్థిరరాశి పూర్ణాంకం p = sector - sections.v[0];
        theta_పునాది = pos[i2 + ne02] * dpct::ఘాతం(theta_scale, p);
    }

    స్థిరరాశి దశాంశం freq_గుణకం = has_ff ? freq_factors[i0 / 2] : 1.0f;

    దశాంశం cos_theta;
    దశాంశం sin_theta;

    rope_yarn<forward>(theta_పునాది / freq_గుణకం, freq_scale, corr_dims, i0,
                       ext_గుణకం, attn_గుణకం, cos_theta, sin_theta);

    స్థిరరాశి దశాంశం x0 = x[ix + 0];
    స్థిరరాశి దశాంశం x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

మూస <తార్కిక forward, రకం_పేరు T, రకం_పేరు D>
స్థిర శూన్య
rope_norm_sycl(స్థిరరాశి T *x, D *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
               స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02, స్థిరరాశి పూర్ణాంకం s03,
               స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3, స్థిరరాశి పూర్ణాంకం n_dims,
               స్థిరరాశి పూర్ణాంకం n_offs, స్థిరరాశి పూర్ణాంకం nr, స్థిరరాశి int32_t *pos, స్థిరరాశి దశాంశం freq_scale,
               స్థిరరాశి దశాంశం freq_పునాది, స్థిరరాశి దశాంశం ext_గుణకం,
               స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
               స్థిరరాశి దశాంశం *freq_factors, స్థిరరాశి int64_t *అడ్డువరుస_indices,
               స్థిరరాశి పూర్ణాంకం సమితి_అడ్డువరుసలు_stride, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    స్థిరరాశి dpct::dim3 ఖండం_dims(1, SYCL_ROPE_ఖండం_కొలత, 1);
    స్థిరరాశి పూర్ణాంకం n_ఖండాలు_x =
        (ne00 + 2 * SYCL_ROPE_ఖండం_కొలత - 1) / (2 * SYCL_ROPE_ఖండం_కొలత);
    స్థిరరాశి dpct::dim3 ఖండం_nums(nr, n_ఖండాలు_x, 1);

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f / n_dims);

    ఒకవేళ (freq_factors == శూన్యసూచిక) {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_norm<forward, అబద్ధం>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride);
            });
    } లేకపోతే {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_norm<forward, నిజం>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride);
            });
    }
}

మూస <తార్కిక forward, రకం_పేరు T, రకం_పేరు D>
స్థిర శూన్య
rope_neox_sycl(స్థిరరాశి T *x, D *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
               స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02, స్థిరరాశి పూర్ణాంకం s03,
               స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3, స్థిరరాశి పూర్ణాంకం n_dims,
               స్థిరరాశి పూర్ణాంకం n_offs, స్థిరరాశి పూర్ణాంకం nr, స్థిరరాశి int32_t *pos, స్థిరరాశి దశాంశం freq_scale,
               స్థిరరాశి దశాంశం freq_పునాది, స్థిరరాశి దశాంశం ext_గుణకం,
               స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
               స్థిరరాశి దశాంశం *freq_factors, స్థిరరాశి int64_t *అడ్డువరుస_indices,
               స్థిరరాశి పూర్ణాంకం సమితి_అడ్డువరుసలు_stride, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    స్థిరరాశి dpct::dim3 ఖండం_dims(1, SYCL_ROPE_ఖండం_కొలత, 1);
    స్థిరరాశి పూర్ణాంకం n_ఖండాలు_x =
        (ne00 + 2 * SYCL_ROPE_ఖండం_కొలత - 1) / (2 * SYCL_ROPE_ఖండం_కొలత);
    స్థిరరాశి dpct::dim3 ఖండం_nums(nr, n_ఖండాలు_x, 1);

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f / n_dims);

    ఒకవేళ (freq_factors == శూన్యసూచిక) {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_neox<forward, అబద్ధం>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride);
            });
    } లేకపోతే {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_neox<forward, నిజం>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride);
            });
    }
}

మూస <తార్కిక forward, రకం_పేరు T>
స్థిర శూన్య
rope_multi_sycl(స్థిరరాశి T *x, T *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
                స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02, స్థిరరాశి పూర్ణాంకం s03,
                స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3, స్థిరరాశి పూర్ణాంకం n_dims,
                స్థిరరాశి పూర్ణాంకం n_offs, స్థిరరాశి పూర్ణాంకం nr, స్థిరరాశి int32_t *pos, స్థిరరాశి దశాంశం freq_scale,
                స్థిరరాశి దశాంశం freq_పునాది, స్థిరరాశి దశాంశం ext_గుణకం,
                స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
                స్థిరరాశి దశాంశం *freq_factors, స్థిరరాశి mrope_sections sections,
                స్థిరరాశి తార్కిక is_imrope, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    స్థిరరాశి dpct::dim3 ఖండం_dims(1, SYCL_ROPE_ఖండం_కొలత, 1);
    స్థిరరాశి పూర్ణాంకం n_ఖండాలు_x =
        (ne00 + 2 * SYCL_ROPE_ఖండం_కొలత - 1) / (2 * SYCL_ROPE_ఖండం_కొలత);
    స్థిరరాశి dpct::dim3 ఖండం_nums(nr, n_ఖండాలు_x, 1);

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f / n_dims);

    ఒకవేళ (freq_factors == శూన్యసూచిక) {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_multi<forward, అబద్ధం, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } లేకపోతే {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_multi<forward, నిజం, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

మూస <తార్కిక forward, రకం_పేరు T>
స్థిర శూన్య
rope_vision_sycl(స్థిరరాశి T *x, T *dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం ne01,
                 స్థిరరాశి పూర్ణాంకం ne02, స్థిరరాశి పూర్ణాంకం s01, స్థిరరాశి పూర్ణాంకం s02, స్థిరరాశి పూర్ణాంకం s03,
                 స్థిరరాశి పూర్ణాంకం s1, స్థిరరాశి పూర్ణాంకం s2, స్థిరరాశి పూర్ణాంకం s3, స్థిరరాశి పూర్ణాంకం n_dims,
                 స్థిరరాశి పూర్ణాంకం nr, స్థిరరాశి int32_t *pos, స్థిరరాశి దశాంశం freq_scale,
                 స్థిరరాశి దశాంశం freq_పునాది, స్థిరరాశి దశాంశం ext_గుణకం,
                 స్థిరరాశి దశాంశం attn_గుణకం, స్థిరరాశి rope_corr_dims corr_dims,
                 స్థిరరాశి దశాంశం *freq_factors, స్థిరరాశి mrope_sections sections,
                 dpct::వరుస_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    స్థిరరాశి dpct::dim3 ఖండం_dims(1, SYCL_ROPE_ఖండం_కొలత, 1);
    స్థిరరాశి పూర్ణాంకం n_ఖండాలు_x =
        (ne00 + 2 * SYCL_ROPE_ఖండం_కొలత - 1) / (2 * SYCL_ROPE_ఖండం_కొలత);
    స్థిరరాశి dpct::dim3 ఖండం_nums(nr, n_ఖండాలు_x, 1);

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f / n_dims);

    ఒకవేళ (freq_factors == శూన్యసూచిక) {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_vision<forward, అబద్ధం, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } లేకపోతే {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                GGML_వాడని(అంశం_ct1);
                rope_vision<forward, నిజం, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_గుణకం, attn_గుణకం, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

మూస <తార్కిక forward>
శూన్య ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            స్థిరరాశి ggml_tensor *సమితి_అడ్డువరుసలు = శూన్యసూచిక) {
    స్థిరరాశి ggml_tensor *src0 = dst->src[0];
    స్థిరరాశి ggml_tensor *src1 = dst->src[1];
    స్థిరరాశి ggml_tensor *src2 = dst->src[2];

    స్థిరరాశి దశాంశం *src0_d = (స్థిరరాశి దశాంశం *)src0->దత్తాంశం;
    స్థిరరాశి దశాంశం *src1_d = (స్థిరరాశి దశాంశం *)src1->దత్తాంశం;

    శూన్య *dst_d = dst->దత్తాంశం;
    స్థిరరాశి int64_t *అడ్డువరుస_indices = శూన్యసూచిక;
    ggml_type dst_type = dst->type;
    పూర్ణాంకం సమితి_అడ్డువరుసలు_stride = 0;

    ఒకవేళ (సమితి_అడ్డువరుసలు != శూన్యసూచిక) {
        GGML_ASSERT(forward);
        dst_d = సమితి_అడ్డువరుసలు->దత్తాంశం;
        అడ్డువరుస_indices = (స్థిరరాశి int64_t *)సమితి_అడ్డువరుసలు->src[1]->దత్తాంశం;
        dst_type = సమితి_అడ్డువరుసలు->type;
        సమితి_అడ్డువరుసలు_stride = సమితి_అడ్డువరుసలు->nb[1] / ggml_type_కొలత(సమితి_అడ్డువరుసలు->type);
    }
    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    స్థిరరాశి int64_t ne00 = src0->ne[0]; // head dims
    స్థిరరాశి int64_t ne01 = src0->ne[1]; // num heads
    స్థిరరాశి int64_t ne02 = src0->ne[2]; // num heads
    స్థిరరాశి int64_t nr = ggml_nrows(src0);

    స్థిరరాశి size_t s01 = src0->nb[1] / ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t s02 = src0->nb[2] / ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t s03 = src0->nb[3] / ggml_type_కొలత(src0->type);

    స్థిరరాశి size_t s1 = dst->nb[1] / ggml_type_కొలత(dst->type);
    స్థిరరాశి size_t s2 = dst->nb[2] / ggml_type_కొలత(dst->type);
    స్థిరరాశి size_t s3 = dst->nb[3] / ggml_type_కొలత(dst->type);

    స్థిరరాశి పూర్ణాంకం n_dims = ((int32_t *)dst->op_params)[1];
    స్థిరరాశి పూర్ణాంకం mode = ((int32_t *)dst->op_params)[2];
    స్థిరరాశి పూర్ణాంకం n_ctx_orig = ((int32_t *)dst->op_params)[4];
    స్థిరరాశి పూర్ణాంకం n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    దశాంశం freq_పునాది;
    దశాంశం freq_scale;
    దశాంశం ext_గుణకం;
    దశాంశం attn_గుణకం;
    దశాంశం beta_fast;
    దశాంశం beta_slow;

    జ్ఞాపకం_నకలు(&freq_పునాది, (int32_t *)dst->op_params + 5, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&freq_scale, (int32_t *)dst->op_params + 6, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&ext_గుణకం, (int32_t *)dst->op_params + 7, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&attn_గుణకం, (int32_t *)dst->op_params + 8, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_fast, (int32_t *)dst->op_params + 9, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_slow, (int32_t *)dst->op_params + 10, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&sections.v, (int32_t *)dst->op_params + 11, పరిమాణం(పూర్ణాంకం) * 4);

    స్థిరరాశి తార్కిక is_neox = mode & GGML_ROPE_TYPE_NEOX;
    స్థిరరాశి తార్కిక is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    స్థిరరాశి తార్కిక is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    స్థిరరాశి తార్కిక is_vision = mode == GGML_ROPE_TYPE_VISION;

    ఒకవేళ (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    ఒకవేళ (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    స్థిరరాశి int32_t *pos = (స్థిరరాశి int32_t *)src1_d;

    స్థిరరాశి దశాంశం *freq_factors = శూన్యసూచిక;
    ఒకవేళ (src2 != శూన్యసూచిక) {
        freq_factors = (స్థిరరాశి దశాంశం *)src2->దత్తాంశం;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    ఒకవేళ (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, దశాంశం, దశాంశం>(
                (స్థిరరాశి దశాంశం *)src0_d, (దశాంశం *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_పునాది,
                ext_గుణకం, attn_గుణకం, corr_dims, freq_factors, అడ్డువరుస_indices,
                సమితి_అడ్డువరుసలు_stride, stream);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, దశాంశం, sycl::half>(
                (స్థిరరాశి దశాంశం *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_పునాది, ext_గుణకం, attn_గుణకం, corr_dims, freq_factors,
                అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride, stream);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (స్థిరరాశి sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_పునాది, ext_గుణకం, attn_గుణకం, corr_dims, freq_factors,
                అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride, stream);
        } లేకపోతే {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } లేకపోతే ఒకవేళ (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        ఒకవేళ (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((స్థిరరాశి దశాంశం *)src0_d, (దశాంశం *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_పునాది,
                                     ext_గుణకం, attn_గుణకం, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (స్థిరరాశి sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_పునాది, ext_గుణకం, attn_గుణకం, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } లేకపోతే {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } లేకపోతే ఒకవేళ (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        ఒకవేళ (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (స్థిరరాశి దశాంశం *)src0_d, (దశాంశం *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_పునాది,
                ext_గుణకం, attn_గుణకం, corr_dims, freq_factors, sections,
                stream);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (స్థిరరాశి sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_పునాది, ext_గుణకం, attn_గుణకం, corr_dims, freq_factors,
                sections, stream);
        } లేకపోతే {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } లేకపోతే {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, దశాంశం, దశాంశం>(
                (స్థిరరాశి దశాంశం *)src0_d, (దశాంశం *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_పునాది,
                ext_గుణకం, attn_గుణకం, corr_dims, freq_factors, అడ్డువరుస_indices,
                సమితి_అడ్డువరుసలు_stride, stream);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, దశాంశం, sycl::half>(
                (స్థిరరాశి దశాంశం *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_పునాది, ext_గుణకం, attn_గుణకం, corr_dims, freq_factors,
                అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride, stream);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (స్థిరరాశి sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_పునాది, ext_గుణకం, attn_గుణకం, corr_dims, freq_factors,
                అడ్డువరుస_indices, సమితి_అడ్డువరుసలు_stride, stream);
        } లేకపోతే {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

శూన్య ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<నిజం>(ctx, dst);
}

శూన్య ggml_sycl_rope_వెనుక(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<అబద్ధం>(ctx, dst);
}

శూన్య ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *సమితి_అడ్డువరుసలు) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<నిజం>(ctx, rope, సమితి_అడ్డువరుసలు);
}
