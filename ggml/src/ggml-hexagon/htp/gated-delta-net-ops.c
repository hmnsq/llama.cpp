#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن <string.h>
#ضمن <math.h>
#ضمن <HAP_farf.h>

#ضمن "hvx-base.h"
#ضمن "hvx-copy.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-exp.h"
#ضمن "dma-queue.h"
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"
#ضمن "hmx-utils.h"
#ضمن "hmx-fa-kernels.h"
#ضمن "hmx-queue.h"
#ضمن "gated-delta-net-ops.h"

#جيڪڏهن_وصف_نه_ٿيل MIN
#وصف_ڪريو MIN(a, b) ((a) < (b) ? (a) : (b))
#جيڪڏهن_پورو

بنية htp_gdn_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_gdn_kernel_params * kparams;
    بنية htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_بنياد;
    uint32_t صف_شروعات;
    uint32_t nrows;
};

ساكن مضمن HVX_ويڪٽر gdn_mul_dot_f32(عائم * محدود dst, ثابت HVX_ويڪٽر * محدود mul, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc = Q6_V_vzero();
    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vd   = hvx_vmemu(dst + i * epv);
        HVX_ويڪٽر vm   = mul[i];
        HVX_ويڪٽر vdot = dot[i];
        HVX_ويڪٽر ٻاهريون  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = ٻاهريون;
        acc = hvx_vec_جوڙيو_f32_f32(acc, hvx_vec_mul_f32_f32(ٻاهريون, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vm = mul[nvec];
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر ٻاهريون = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * حجم(عائم), ٻاهريون);
        acc = hvx_vec_جوڙيو_f32_f32(acc, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(ٻاهريون, vdot), zero));
    }

    أرجع hvx_vec_reduce_جوڙ_f32(acc);
}

ساكن مضمن HVX_ويڪٽر gdn_mul_scalar_dot_f32(عائم * محدود dst, HVX_ويڪٽر vmul, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc = Q6_V_vzero();
    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vd   = hvx_vmemu(dst + i * epv);
        HVX_ويڪٽر vdot = dot[i];
        HVX_ويڪٽر ٻاهريون  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = ٻاهريون;
        acc = hvx_vec_جوڙيو_f32_f32(acc, hvx_vec_mul_f32_f32(ٻاهريون, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر ٻاهريون = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * حجم(عائم), ٻاهريون);
        acc = hvx_vec_جوڙيو_f32_f32(acc, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(ٻاهريون, vdot), zero));
    }

    أرجع hvx_vec_reduce_جوڙ_f32(acc);
}

ساكن مضمن HVX_ويڪٽر gdn_جوڙيو_scaled_dot_f32(عائم * محدود dst, ثابت HVX_ويڪٽر * محدود src,
        HVX_ويڪٽر vscale, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc = Q6_V_vzero();
    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vd   = hvx_vmemu(dst + i * epv);
        HVX_ويڪٽر vs   = src[i];
        HVX_ويڪٽر vdot = dot[i];
        HVX_ويڪٽر ٻاهريون  = hvx_vec_جوڙيو_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = ٻاهريون;
        acc = hvx_vec_جوڙيو_f32_f32(acc, hvx_vec_mul_f32_f32(ٻاهريون, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vs   = src[nvec];
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر ٻاهريون = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * حجم(عائم), ٻاهريون);
        acc = hvx_vec_جوڙيو_f32_f32(acc, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(ٻاهريون, vdot), zero));
    }

    أرجع hvx_vec_reduce_جوڙ_f32(acc);
}

ساكن مضمن HVX_ويڪٽر gdn_mul_dot4_f32(عائم * محدود dst0, عائم * محدود dst1,
        عائم * محدود dst2, عائم * محدود dst3,
        ثابت HVX_ويڪٽر * محدود mul, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc0 = Q6_V_vzero();
    HVX_ويڪٽر acc1 = Q6_V_vzero();
    HVX_ويڪٽر acc2 = Q6_V_vzero();
    HVX_ويڪٽر acc3 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vm   = mul[i];
        HVX_ويڪٽر vdot = dot[i];

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vm = mul[nvec];
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * حجم(عائم), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(عائم), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(عائم), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(عائم), out3);

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ويڪٽر_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    أرجع hvx_vec_reduce_جوڙ_f32x4(acc);
}

ساكن مضمن HVX_ويڪٽر gdn_mul_scalar_dot4_f32(عائم * محدود dst0, عائم * محدود dst1,
        عائم * محدود dst2, عائم * محدود dst3,
        HVX_ويڪٽر vmul, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc0 = Q6_V_vzero();
    HVX_ويڪٽر acc1 = Q6_V_vzero();
    HVX_ويڪٽر acc2 = Q6_V_vzero();
    HVX_ويڪٽر acc3 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vdot = dot[i];

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * حجم(عائم), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(عائم), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(عائم), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(عائم), out3);

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ويڪٽر_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    أرجع hvx_vec_reduce_جوڙ_f32x4(acc);
}

ساكن مضمن HVX_ويڪٽر gdn_جوڙيو_scaled_dot4_f32(عائم * محدود dst0, عائم * محدود dst1,
        عائم * محدود dst2, عائم * محدود dst3,
        ثابت HVX_ويڪٽر * محدود src, ثابت عائم * محدود scale,
        ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc0 = Q6_V_vzero();
    HVX_ويڪٽر acc1 = Q6_V_vzero();
    HVX_ويڪٽر acc2 = Q6_V_vzero();
    HVX_ويڪٽر acc3 = Q6_V_vzero();
    ثابت HVX_ويڪٽر scale0 = hvx_vec_splat_f32(scale[0]);
    ثابت HVX_ويڪٽر scale1 = hvx_vec_splat_f32(scale[1]);
    ثابت HVX_ويڪٽر scale2 = hvx_vec_splat_f32(scale[2]);
    ثابت HVX_ويڪٽر scale3 = hvx_vec_splat_f32(scale[3]);

    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vs   = src[i];
        HVX_ويڪٽر vdot = dot[i];

        HVX_ويڪٽر out0 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ويڪٽر out1 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ويڪٽر out2 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ويڪٽر out3 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vs   = src[nvec];
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر out0 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ويڪٽر out1 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ويڪٽر out2 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ويڪٽر out3 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * حجم(عائم), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(عائم), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(عائم), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(عائم), out3);

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ويڪٽر_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    أرجع hvx_vec_reduce_جوڙ_f32x4(acc);
}

ساكن مضمن HVX_ويڪٽر gdn_mul_dot8_f32(عائم * محدود dst0, عائم * محدود dst1,
        عائم * محدود dst2, عائم * محدود dst3, عائم * محدود dst4,
        عائم * محدود dst5, عائم * محدود dst6, عائم * محدود dst7,
        ثابت HVX_ويڪٽر * محدود mul, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc0 = Q6_V_vzero();
    HVX_ويڪٽر acc1 = Q6_V_vzero();
    HVX_ويڪٽر acc2 = Q6_V_vzero();
    HVX_ويڪٽر acc3 = Q6_V_vzero();
    HVX_ويڪٽر acc4 = Q6_V_vzero();
    HVX_ويڪٽر acc5 = Q6_V_vzero();
    HVX_ويڪٽر acc6 = Q6_V_vzero();
    HVX_ويڪٽر acc7 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vm   = mul[i];
        HVX_ويڪٽر vdot = dot[i];

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_ويڪٽر out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_ويڪٽر out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_ويڪٽر out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_ويڪٽر out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_جوڙيو_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_جوڙيو_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_جوڙيو_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_جوڙيو_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vm = mul[nvec];
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_ويڪٽر out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_ويڪٽر out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_ويڪٽر out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_ويڪٽر out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * حجم(عائم), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(عائم), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(عائم), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(عائم), out3);
        hvx_vec_store_u(dst4 + off, nloe * حجم(عائم), out4);
        hvx_vec_store_u(dst5 + off, nloe * حجم(عائم), out5);
        hvx_vec_store_u(dst6 + off, nloe * حجم(عائم), out6);
        hvx_vec_store_u(dst7 + off, nloe * حجم(عائم), out7);

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_جوڙيو_f32_f32(acc4, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_جوڙيو_f32_f32(acc5, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_جوڙيو_f32_f32(acc6, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_جوڙيو_f32_f32(acc7, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ويڪٽر_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ويڪٽر_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ويڪٽر rA = hvx_vec_reduce_جوڙ_f32x4(accA);
    HVX_ويڪٽر rB = hvx_vec_reduce_جوڙ_f32x4(accB);
    HVX_ويڪٽر_Pred q16 = Q6_Q_vsetq2_R(16);
    أرجع Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ساكن مضمن HVX_ويڪٽر gdn_mul_scalar_dot8_f32(عائم * محدود dst0, عائم * محدود dst1,
        عائم * محدود dst2, عائم * محدود dst3, عائم * محدود dst4,
        عائم * محدود dst5, عائم * محدود dst6, عائم * محدود dst7,
        HVX_ويڪٽر vmul, ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc0 = Q6_V_vzero();
    HVX_ويڪٽر acc1 = Q6_V_vzero();
    HVX_ويڪٽر acc2 = Q6_V_vzero();
    HVX_ويڪٽر acc3 = Q6_V_vzero();
    HVX_ويڪٽر acc4 = Q6_V_vzero();
    HVX_ويڪٽر acc5 = Q6_V_vzero();
    HVX_ويڪٽر acc6 = Q6_V_vzero();
    HVX_ويڪٽر acc7 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vdot = dot[i];

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_ويڪٽر out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_ويڪٽر out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_ويڪٽر out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_ويڪٽر out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_جوڙيو_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_جوڙيو_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_جوڙيو_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_جوڙيو_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_ويڪٽر out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_ويڪٽر out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_ويڪٽر out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_ويڪٽر out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_ويڪٽر out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_ويڪٽر out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_ويڪٽر out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * حجم(عائم), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(عائم), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(عائم), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(عائم), out3);
        hvx_vec_store_u(dst4 + off, nloe * حجم(عائم), out4);
        hvx_vec_store_u(dst5 + off, nloe * حجم(عائم), out5);
        hvx_vec_store_u(dst6 + off, nloe * حجم(عائم), out6);
        hvx_vec_store_u(dst7 + off, nloe * حجم(عائم), out7);

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_جوڙيو_f32_f32(acc4, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_جوڙيو_f32_f32(acc5, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_جوڙيو_f32_f32(acc6, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_جوڙيو_f32_f32(acc7, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ويڪٽر_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ويڪٽر_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ويڪٽر rA = hvx_vec_reduce_جوڙ_f32x4(accA);
    HVX_ويڪٽر rB = hvx_vec_reduce_جوڙ_f32x4(accB);
    HVX_ويڪٽر_Pred q16 = Q6_Q_vsetq2_R(16);
    أرجع Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ساكن مضمن HVX_ويڪٽر gdn_جوڙيو_scaled_dot8_f32(عائم * محدود dst0, عائم * محدود dst1,
        عائم * محدود dst2, عائم * محدود dst3, عائم * محدود dst4,
        عائم * محدود dst5, عائم * محدود dst6, عائم * محدود dst7,
        ثابت HVX_ويڪٽر * محدود src, ثابت عائم * محدود scale,
        ثابت HVX_ويڪٽر * محدود dot, uint32_t n) {
    HVX_ويڪٽر acc0 = Q6_V_vzero();
    HVX_ويڪٽر acc1 = Q6_V_vzero();
    HVX_ويڪٽر acc2 = Q6_V_vzero();
    HVX_ويڪٽر acc3 = Q6_V_vzero();
    HVX_ويڪٽر acc4 = Q6_V_vzero();
    HVX_ويڪٽر acc5 = Q6_V_vzero();
    HVX_ويڪٽر acc6 = Q6_V_vzero();
    HVX_ويڪٽر acc7 = Q6_V_vzero();
    ثابت HVX_ويڪٽر scale0 = hvx_vec_splat_f32(scale[0]);
    ثابت HVX_ويڪٽر scale1 = hvx_vec_splat_f32(scale[1]);
    ثابت HVX_ويڪٽر scale2 = hvx_vec_splat_f32(scale[2]);
    ثابت HVX_ويڪٽر scale3 = hvx_vec_splat_f32(scale[3]);
    ثابت HVX_ويڪٽر scale4 = hvx_vec_splat_f32(scale[4]);
    ثابت HVX_ويڪٽر scale5 = hvx_vec_splat_f32(scale[5]);
    ثابت HVX_ويڪٽر scale6 = hvx_vec_splat_f32(scale[6]);
    ثابت HVX_ويڪٽر scale7 = hvx_vec_splat_f32(scale[7]);

    ثابت uint32_t epv = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    لكل (uint32_t i = 0; i < nvec; ++i) {
        HVX_ويڪٽر vs   = src[i];
        HVX_ويڪٽر vdot = dot[i];

        HVX_ويڪٽر out0 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ويڪٽر out1 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ويڪٽر out2 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ويڪٽر out3 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_ويڪٽر out4 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_ويڪٽر out5 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_ويڪٽر out6 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_ويڪٽر out7 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_جوڙيو_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_جوڙيو_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_جوڙيو_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_جوڙيو_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    إذا (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_ويڪٽر vs   = src[nvec];
        HVX_ويڪٽر vdot = dot[nvec];
        HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(عائم));
        HVX_ويڪٽر zero = Q6_V_vzero();

        HVX_ويڪٽر out0 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ويڪٽر out1 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ويڪٽر out2 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ويڪٽر out3 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_ويڪٽر out4 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_ويڪٽر out5 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_ويڪٽر out6 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_ويڪٽر out7 = hvx_vec_جوڙيو_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * حجم(عائم), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(عائم), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(عائم), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(عائم), out3);
        hvx_vec_store_u(dst4 + off, nloe * حجم(عائم), out4);
        hvx_vec_store_u(dst5 + off, nloe * حجم(عائم), out5);
        hvx_vec_store_u(dst6 + off, nloe * حجم(عائم), out6);
        hvx_vec_store_u(dst7 + off, nloe * حجم(عائم), out7);

        acc0 = hvx_vec_جوڙيو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڙيو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڙيو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڙيو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_جوڙيو_f32_f32(acc4, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_جوڙيو_f32_f32(acc5, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_جوڙيو_f32_f32(acc6, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_جوڙيو_f32_f32(acc7, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ويڪٽر_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ويڪٽر_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ويڪٽر rA = hvx_vec_reduce_جوڙ_f32x4(accA);
    HVX_ويڪٽر rB = hvx_vec_reduce_جوڙ_f32x4(accB);
    HVX_ويڪٽر_Pred q16 = Q6_Q_vsetq2_R(16);
    أرجع Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ساكن مضمن فراغ gdn_قدم_kda_f32(
    عائم * محدود s_work,
    عائم * محدود attn_ٻاهريون,
    ثابت عائم * محدود q_t,
    ثابت عائم * محدود k_t,
    ثابت عائم * محدود v_t,
    ثابت عائم * محدود g_t,
    عائم beta_val,
    عائم scale,
    uint32_t S_v
) {
    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = S_v / epv;
    ثابت uint32_t nloe = S_v % epv;

    HVX_ويڪٽر vq[4];
    HVX_ويڪٽر vk[4];
    HVX_ويڪٽر vg[4];

    لكل (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    إذا (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
    }

    ثابت HVX_ويڪٽر vbeta  = hvx_vec_splat_f32(beta_val);
    ثابت HVX_ويڪٽر vscale = hvx_vec_splat_f32(scale);

    عائم delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    لكل (; j + 8 <= S_v; j += 8) {
        عائم * row0 = s_work + (uint64_t) (j + 0) * S_v;
        عائم * row1 = s_work + (uint64_t) (j + 1) * S_v;
        عائم * row2 = s_work + (uint64_t) (j + 2) * S_v;
        عائم * row3 = s_work + (uint64_t) (j + 3) * S_v;
        عائم * row4 = s_work + (uint64_t) (j + 4) * S_v;
        عائم * row5 = s_work + (uint64_t) (j + 5) * S_v;
        عائم * row6 = s_work + (uint64_t) (j + 6) * S_v;
        عائم * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_ويڪٽر vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_ويڪٽر vv_t   = hvx_vmemu(v_t + j);
        HVX_ويڪٽر diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ويڪٽر vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * حجم(عائم), vdelta);

        HVX_ويڪٽر vattn = gdn_جوڙيو_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_ويڪٽر res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ٻاهريون + j, 8 * حجم(عائم), res_attn);
    }
    لكل (; j + 4 <= S_v; j += 4) {
        عائم * row0 = s_work + (uint64_t) (j + 0) * S_v;
        عائم * row1 = s_work + (uint64_t) (j + 1) * S_v;
        عائم * row2 = s_work + (uint64_t) (j + 2) * S_v;
        عائم * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_ويڪٽر vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_ويڪٽر vv_t   = hvx_vmemu(v_t + j);
        HVX_ويڪٽر diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ويڪٽر vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * حجم(عائم), vdelta);

        HVX_ويڪٽر vattn = gdn_جوڙيو_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_ويڪٽر res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ٻاهريون + j, 4 * حجم(عائم), res_attn);
    }
    لكل (; j < S_v; ++j) {
        عائم * صف = s_work + (uint64_t) j * S_v;
        HVX_ويڪٽر vsum = gdn_mul_dot_f32(صف, vg, vk, S_v);
        HVX_ويڪٽر vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_ويڪٽر vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_ويڪٽر vres = gdn_جوڙيو_scaled_dot_f32(صف, vk, vdj, vq, S_v);
        attn_ٻاهريون[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ساكن مضمن فراغ gdn_قدم_scalar_f32(
    عائم * محدود s_work,
    عائم * محدود attn_ٻاهريون,
    ثابت عائم * محدود q_t,
    ثابت عائم * محدود k_t,
    ثابت عائم * محدود v_t,
    ثابت عائم * محدود g_t,
    عائم beta_val,
    عائم scale,
    uint32_t S_v
) {
    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = S_v / epv;
    ثابت uint32_t nloe = S_v % epv;

    HVX_ويڪٽر vq[4];
    HVX_ويڪٽر vk[4];

    لكل (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    إذا (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    ثابت HVX_ويڪٽر vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    ثابت HVX_ويڪٽر vbeta  = hvx_vec_splat_f32(beta_val);
    ثابت HVX_ويڪٽر vscale = hvx_vec_splat_f32(scale);

    عائم delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    لكل (; j + 8 <= S_v; j += 8) {
        عائم * row0 = s_work + (uint64_t) (j + 0) * S_v;
        عائم * row1 = s_work + (uint64_t) (j + 1) * S_v;
        عائم * row2 = s_work + (uint64_t) (j + 2) * S_v;
        عائم * row3 = s_work + (uint64_t) (j + 3) * S_v;
        عائم * row4 = s_work + (uint64_t) (j + 4) * S_v;
        عائم * row5 = s_work + (uint64_t) (j + 5) * S_v;
        عائم * row6 = s_work + (uint64_t) (j + 6) * S_v;
        عائم * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_ويڪٽر vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_ويڪٽر vv_t   = hvx_vmemu(v_t + j);
        HVX_ويڪٽر diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ويڪٽر vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * حجم(عائم), vdelta);

        HVX_ويڪٽر vattn = gdn_جوڙيو_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_ويڪٽر res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ٻاهريون + j, 8 * حجم(عائم), res_attn);
    }
    لكل (; j + 4 <= S_v; j += 4) {
        عائم * row0 = s_work + (uint64_t) (j + 0) * S_v;
        عائم * row1 = s_work + (uint64_t) (j + 1) * S_v;
        عائم * row2 = s_work + (uint64_t) (j + 2) * S_v;
        عائم * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_ويڪٽر vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_ويڪٽر vv_t   = hvx_vmemu(v_t + j);
        HVX_ويڪٽر diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ويڪٽر vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * حجم(عائم), vdelta);

        HVX_ويڪٽر vattn = gdn_جوڙيو_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_ويڪٽر res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ٻاهريون + j, 4 * حجم(عائم), res_attn);
    }
    لكل (; j < S_v; ++j) {
        عائم * صف = s_work + (uint64_t) j * S_v;
        HVX_ويڪٽر vsum = gdn_mul_scalar_dot_f32(صف, vgate, vk, S_v);
        HVX_ويڪٽر vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_ويڪٽر vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_ويڪٽر vres = gdn_جوڙيو_scaled_dot_f32(صف, vk, vdj, vq, S_v);
        attn_ٻاهريون[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ساكن فراغ gated_delta_net_f32_pp_thread(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_gdn_context * gctx = (بنية htp_gdn_context *) معلومات;
    بنية htp_ops_context * octx = gctx->octx;
    ثابت بنية htp_gdn_kernel_params * kparams = gctx->kparams;

    ثابت بنية htp_tensor * q     = octx->src[0];
    ثابت بنية htp_tensor * k     = octx->src[1];
    ثابت بنية htp_tensor * v     = octx->src[2];
    ثابت بنية htp_tensor * g     = octx->src[3];
    ثابت بنية htp_tensor * beta  = octx->src[4];
    ثابت بنية htp_tensor * state = octx->src[5];
    ثابت بنية htp_tensor * dst   = octx->dst;

    ثابت uint32_t S_v      = kparams->S_v;
    ثابت uint32_t H        = kparams->H;
    ثابت uint32_t n_ٽوڪن = kparams->n_ٽوڪن;
    ثابت uint32_t n_seqs   = kparams->n_seqs;
    ثابت uint32_t K        = kparams->K;
    ثابت uint32_t صف_پڇاڙي  = gctx->صف_شروعات + gctx->nrows;

    إذا (ith >= gctx->nrows) {
        أرجع;
    }

    ثابت بنية htp_tensor * dst_cache = octx->dsts[1];
    ثابت عائم scale = kparams->scale;
    عائم * dst_بنياد       = (عائم *) (uintptr_t) dst->معلومات;
    عائم * state_ٻاهريون_بنياد = dst_cache ? (عائم *) (uintptr_t) dst_cache->معلومات : (dst_بنياد + S_v * H * n_ٽوڪن * n_seqs);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ثابت بنية htp_gdn_vtcm_layout * layout = &gctx->layout;
    عائم * s_work[2];
    s_work[0] = (عائم *) (gctx->vtcm_بنياد + layout->بائيٽون_في_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / حجم(عائم);

    ثابت بنية fastdiv_قدرون * fd_H   = &kparams->div_H;
    ثابت بنية fastdiv_قدرون * fd_q1  = &kparams->div_q1;
    ثابت بنية fastdiv_قدرون * fd_k1  = &kparams->div_k1;
    ثابت بنية fastdiv_قدرون * fd_rq3 = &kparams->div_rq3;
    ثابت بنية fastdiv_قدرون * fd_rk3 = &kparams->div_rk3;

    ثابت uint32_t state_seq_stride = kparams->state_seq_stride;
    ثابت uint64_t state_سائيز_في_snap = (uint64_t) kparams->state_سائيز_في_snap;
    ثابت dma_addr_t state_ٻاهريون_dma_بنياد = dst_cache ? dst_cache->معلومات : (dst->معلومات + S_v * H * n_ٽوڪن * n_seqs * حجم(عائم));

    uint32_t ir_prefetch = gctx->صف_شروعات + ith;
    صحيح spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    لكل (صحيح قدم = 0; قدم < 2 && ir_prefetch < صف_پڇاڙي; قدم++) {
        ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_اندريون  = state->معلومات + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(عائم);
        dma_addr_t ps_ٻاهريون = state_ٻاهريون_dma_بنياد + ((uint64_t) piv3 * H + piv1) * S_v * S_v * حجم(عائم);

        // Push dummy write-back
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(ps_ٻاهريون, s_work[spad_idx]),
                       S_v * حجم(عائم), S_v * حجم(عائم),
                       S_v * حجم(عائم), 0);

        // Push fetch
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_work[spad_idx], ps_اندريون),
                       S_v * حجم(عائم), S_v * حجم(عائم),
                       S_v * حجم(عائم), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    صحيح curr_spad_idx = 0;
    لكل (uint32_t ir = gctx->صف_شروعات + ith; ir < صف_پڇاڙي; ir += nth) {
        dma_قطار_ڪڍو(dma_q);
        dma_قطار_ڪڍو(dma_q);

        عائم * s_work_curr = s_work[curr_spad_idx];

        ثابت uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ثابت uint32_t iv3 = fastdiv(ir, fd_H);

        ثابت uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ثابت uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ثابت uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ثابت uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_ٻاهريون  = state_ٻاهريون_dma_بنياد + ((uint64_t) iv3 * H + iv1) * S_v * S_v * حجم(عائم);
        عائم * attn_معلومات = dst_بنياد + ((uint64_t) iv3 * n_ٽوڪن * H + iv1) * S_v;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        لكل (uint32_t t = 0; t < n_ٽوڪن; ++t) {
            ثابت عائم * q_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) q->معلومات +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            ثابت عائم * k_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) k->معلومات +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            ثابت عائم * v_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) v->معلومات +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            ثابت عائم * g_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) g->معلومات +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            ثابت عائم beta_val = *(ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) beta->معلومات +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            إذا (kparams->kda) {
                gdn_قدم_kda_f32(s_work_curr, attn_معلومات, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } وإلا {
                gdn_قدم_scalar_f32(s_work_curr, attn_معلومات, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            إذا (K > 1) {
                ثابت int64_t هدف_جڳهه = (int64_t) n_ٽوڪن - 1 - (int64_t) t;
                إذا (هدف_جڳهه > 0 && هدف_جڳهه < (int64_t) K) {
                    عائم * curr_state_o = state_ٻاهريون_بنياد + (uint64_t) هدف_جڳهه * state_سائيز_في_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (ثابت uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_معلومات += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_ٻاهريون, s_work_curr),
                       S_v * حجم(عائم), S_v * حجم(عائم),
                       S_v * حجم(عائم), S_v);

        // Prefetch next block (if any)
        إذا (ir_prefetch < صف_پڇاڙي) {
            ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_اندريون = state->معلومات + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(عائم);

            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_work[spad_idx], ps_اندريون),
                           S_v * حجم(عائم), S_v * حجم(عائم),
                           S_v * حجم(عائم), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_قطار_flush(dma_q);
}

ساكن فراغ gated_delta_net_f32_tg_thread(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_gdn_context * gctx = (بنية htp_gdn_context *) معلومات;
    بنية htp_ops_context * octx = gctx->octx;
    ثابت بنية htp_gdn_kernel_params * kparams = gctx->kparams;

    ثابت بنية htp_tensor * q     = octx->src[0];
    ثابت بنية htp_tensor * k     = octx->src[1];
    ثابت بنية htp_tensor * v     = octx->src[2];
    ثابت بنية htp_tensor * g     = octx->src[3];
    ثابت بنية htp_tensor * beta  = octx->src[4];
    ثابت بنية htp_tensor * state = octx->src[5];
    ثابت بنية htp_tensor * dst   = octx->dst;

    ثابت uint32_t S_v      = kparams->S_v;
    ثابت uint32_t H        = kparams->H;
    ثابت uint32_t n_seqs   = kparams->n_seqs;
    ثابت uint32_t صف_پڇاڙي  = gctx->صف_شروعات + gctx->nrows;

    إذا (ith >= gctx->nrows) {
        أرجع;
    }

    ثابت بنية htp_tensor * dst_cache = octx->dsts[1];
    ثابت عائم scale = kparams->scale;
    عائم * dst_بنياد  = (عائم *) (uintptr_t) dst->معلومات;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ثابت بنية htp_gdn_vtcm_layout * layout = &gctx->layout;
    عائم * s_work[2];
    s_work[0] = (عائم *) (gctx->vtcm_بنياد + layout->بائيٽون_في_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / حجم(عائم);

    ثابت بنية fastdiv_قدرون * fd_H   = &kparams->div_H;
    ثابت بنية fastdiv_قدرون * fd_q1  = &kparams->div_q1;
    ثابت بنية fastdiv_قدرون * fd_k1  = &kparams->div_k1;
    ثابت بنية fastdiv_قدرون * fd_rq3 = &kparams->div_rq3;
    ثابت بنية fastdiv_قدرون * fd_rk3 = &kparams->div_rk3;

    ثابت uint32_t state_seq_stride = kparams->state_seq_stride;
    ثابت dma_addr_t state_ٻاهريون_dma_بنياد = dst_cache ? dst_cache->معلومات : (dst->معلومات + S_v * H * n_seqs * حجم(عائم));

    uint32_t ir_prefetch = gctx->صف_شروعات + ith;
    صحيح spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    لكل (صحيح قدم = 0; قدم < 2 && ir_prefetch < صف_پڇاڙي; قدم++) {
        ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_اندريون  = state->معلومات + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(عائم);
        dma_addr_t ps_ٻاهريون = state_ٻاهريون_dma_بنياد + ((uint64_t) piv3 * H + piv1) * S_v * S_v * حجم(عائم);

        // Push dummy write-back
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(ps_ٻاهريون, s_work[spad_idx]),
                       S_v * حجم(عائم), S_v * حجم(عائم),
                       S_v * حجم(عائم), 0);

        // Push fetch
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_work[spad_idx], ps_اندريون),
                       S_v * حجم(عائم), S_v * حجم(عائم),
                       S_v * حجم(عائم), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    صحيح curr_spad_idx = 0;
    لكل (uint32_t ir = gctx->صف_شروعات + ith; ir < صف_پڇاڙي; ir += nth) {
        dma_قطار_ڪڍو(dma_q);
        dma_قطار_ڪڍو(dma_q);

        عائم * s_work_curr = s_work[curr_spad_idx];

        ثابت uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ثابت uint32_t iv3 = fastdiv(ir, fd_H);

        ثابت uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ثابت uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ثابت uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ثابت uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_ٻاهريون  = state_ٻاهريون_dma_بنياد + ((uint64_t) iv3 * H + iv1) * S_v * S_v * حجم(عائم);
        عائم * attn_معلومات = dst_بنياد + ((uint64_t) iv3 * H + iv1) * S_v;

        ثابت عائم * q_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) q->معلومات +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        ثابت عائم * k_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) k->معلومات +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        ثابت عائم * v_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) v->معلومات +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        ثابت عائم * g_t = (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) g->معلومات +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        ثابت عائم beta_val = *(ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) beta->معلومات +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (kparams->kda) {
            gdn_قدم_kda_f32(s_work_curr, attn_معلومات, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } وإلا {
            gdn_قدم_scalar_f32(s_work_curr, attn_معلومات, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_ٻاهريون, s_work_curr),
                       S_v * حجم(عائم), S_v * حجم(عائم),
                       S_v * حجم(عائم), S_v);

        // Prefetch next block (if any)
        إذا (ir_prefetch < صف_پڇاڙي) {
            ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_اندريون = state->معلومات + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(عائم);

            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_work[spad_idx], ps_اندريون),
                           S_v * حجم(عائم), S_v * حجم(عائم),
                           S_v * حجم(عائم), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_قطار_flush(dma_q);
}

بنية htp_gdn_hmx_gemm_task {
    ثابت __fp16 * صف_tiles;
    ثابت __fp16 * کانو_tiles;
    __fp16 *       ٻاهريون_tiles;
    uint32_t       n_صف_tiles;
    uint32_t       n_کانو_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

ساكن فراغ htp_gdn_hmx_gemm_ڪم_ڪندڙ(فراغ * معلومات) {
    بنية htp_gdn_hmx_gemm_task * task = (بنية htp_gdn_hmx_gemm_task *) معلومات;
    asm متطاير(HMX_سيٽ_BIAS("%0") :: "r"((بنانشان صحيح)task->hmx_scales));

    ثابت size_t dot_stride = task->dot_stride;
    لكل (uint32_t r = 0; r < task->n_صف_tiles; ++r) {
        ثابت __fp16 * r_tiles = task->صف_tiles + r * dot_stride;
        ثابت __fp16 * c_tiles = task->کانو_tiles;
        __fp16 *       o_tile  = task->ٻاهريون_tiles + r * task->n_کانو_tiles * HMX_FP16_TILE_N_ELMS;

        لكل (uint32_t c = 0; c < task->n_کانو_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

ساكن مضمن فراغ htp_gdn_ڌڪو_hmx_gemm_task(
    hmx_قطار_t q,
    بنية htp_gdn_hmx_gemm_task * task,
    ثابت __fp16 * صف_tiles,
    ثابت __fp16 * کانو_tiles,
    __fp16 * ٻاهريون_tiles,
    uint32_t n_صف_tiles,
    uint32_t n_کانو_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    task->صف_tiles   = صف_tiles;
    task->کانو_tiles   = کانو_tiles;
    task->ٻاهريون_tiles   = ٻاهريون_tiles;
    task->n_صف_tiles = n_صف_tiles;
    task->n_کانو_tiles = n_کانو_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    task->hmx_scales  = scales;

    hmx_قطار_ڌڪو(q, hmx_قطار_make_desc(htp_gdn_hmx_gemm_ڪم_ڪندڙ, task));
}

ساكن مضمن فراغ gdn_unpack_64x64_tiles_منزل_vectors(
    HVX_ويڪٽر * محدود صفون,
    ثابت __fp16 * محدود tiles
) {
    ثابت HVX_ويڪٽر * t00 = (ثابت HVX_ويڪٽر *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    ثابت HVX_ويڪٽر * t01 = (ثابت HVX_ويڪٽر *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    ثابت HVX_ويڪٽر * t10 = (ثابت HVX_ويڪٽر *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    ثابت HVX_ويڪٽر * t11 = (ثابت HVX_ويڪٽر *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    لكل (uint32_t r = 0; r < 16; ++r) {
        HVX_ويڪٽر_Pair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        صفون[2 * r + 0] = Q6_V_lo_W(vp0);
        صفون[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_ويڪٽر_Pair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        صفون[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        صفون[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

ساكن مضمن فراغ gdn_pack_64x64_vectors_منزل_tiles(
    __fp16 * محدود tiles,
    ثابت HVX_ويڪٽر * محدود صفون
) {
    HVX_ويڪٽر * t00 = (HVX_ويڪٽر *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_ويڪٽر * t01 = (HVX_ويڪٽر *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_ويڪٽر * t10 = (HVX_ويڪٽر *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_ويڪٽر * t11 = (HVX_ويڪٽر *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    لكل (uint32_t r = 0; r < 16; ++r) {
        HVX_ويڪٽر_Pair vp0 = Q6_W_vshuff_VVR(صفون[2 * r + 1], صفون[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_ويڪٽر_Pair vp1 = Q6_W_vshuff_VVR(صفون[32 + 2 * r + 1], صفون[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

ساكن مضمن فراغ gdn_unpack_64x_S_tiles_منزل_f32(
    عائم * محدود dst_f32,
    ثابت __fp16 * محدود tiles,
    uint32_t S_v
) {
    ثابت uint32_t n_کانو_tiles = S_v / 32;
    لكل (uint32_t r0 = 0; r0 < 2; ++r0) {
        لكل (uint32_t d = 0; d < S_v / 64; ++d) {
            ثابت HVX_ويڪٽر * t0 = (ثابت HVX_ويڪٽر *) (tiles + (r0 * n_کانو_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ثابت HVX_ويڪٽر * t1 = (ثابت HVX_ويڪٽر *) (tiles + (r0 * n_کانو_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            لكل (uint32_t r = 0; r < 16; ++r) {
                HVX_ويڪٽر_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_ويڪٽر_Pair p0 = hvx_vec_f16_منزل_f32(Q6_V_lo_W(vp01));
                HVX_ويڪٽر_Pair p1 = hvx_vec_f16_منزل_f32(Q6_V_hi_W(vp01));

                عائم * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                عائم * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ساكن مضمن فراغ gdn_unpack_64x_S_tiles_منزل_f16(
    __fp16 * محدود dst_f16,
    ثابت __fp16 * محدود tiles,
    uint32_t S_v
) {
    ثابت uint32_t n_کانو_tiles = S_v / 32;
    لكل (uint32_t r0 = 0; r0 < 2; ++r0) {
        لكل (uint32_t d = 0; d < S_v / 64; ++d) {
            ثابت HVX_ويڪٽر * t0 = (ثابت HVX_ويڪٽر *) (tiles + (r0 * n_کانو_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ثابت HVX_ويڪٽر * t1 = (ثابت HVX_ويڪٽر *) (tiles + (r0 * n_کانو_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            لكل (uint32_t r = 0; r < 16; ++r) {
                HVX_ويڪٽر_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ساكن مضمن فراغ gdn_unpack_Sx_S_tiles_منزل_f32(
    عائم * محدود dst_f32,
    ثابت __fp16 * محدود tiles,
    uint32_t S_v
) {
    ثابت uint32_t n_tiles = S_v / 32;
    لكل (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        لكل (uint32_t d = 0; d < S_v / 64; ++d) {
            ثابت HVX_ويڪٽر * t0 = (ثابت HVX_ويڪٽر *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ثابت HVX_ويڪٽر * t1 = (ثابت HVX_ويڪٽر *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            لكل (uint32_t r = 0; r < 16; ++r) {
                HVX_ويڪٽر_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_ويڪٽر_Pair p0 = hvx_vec_f16_منزل_f32(Q6_V_lo_W(vp01));
                HVX_ويڪٽر_Pair p1 = hvx_vec_f16_منزل_f32(Q6_V_hi_W(vp01));

                عائم * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                عائم * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ساكن مضمن فراغ gdn_f32_منزل_hmx_صف_tiles_and_f16(
    __fp16 * محدود dst_tiles,
    __fp16 * محدود dst_مفرد_tiles,
    __fp16 * محدود dst_f16,
    ثابت عائم * محدود src,
    ثابت __fp16 * محدود scale_في_صف,
    uint32_t n_صفون,
    uint32_t n_کانن_ڳڻپ
) {
    ثابت uint32_t n_کانو_tiles = n_کانن_ڳڻپ / 32;
    ثابت uint32_t * scale_جوڙا = (ثابت uint32_t *) scale_في_صف;

    لكل (uint32_t r = 0; r < n_صفون; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        ثابت عائم * p0 = src + (r + 0) * n_کانن_ڳڻپ;
        ثابت عائم * p1 = src + (r + 1) * n_کانن_ڳڻپ;

        HVX_ويڪٽر v_scale;
        إذا (dst_مفرد_tiles) {
            uint32_t scale_pair = scale_جوڙا ? scale_جوڙا[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        لكل (uint32_t c = 0; c < n_کانو_tiles; c += 2) {
            HVX_ويڪٽر v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_ويڪٽر v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_ويڪٽر v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_ويڪٽر v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_ويڪٽر vh0 = hvx_vec_f32_منزل_f16_shuff(v0_0, v1_0);
            HVX_ويڪٽر vh1 = hvx_vec_f32_منزل_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_کانو_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_کانو_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_ويڪٽر *) tile0)[r1] = vh0;
            ((HVX_ويڪٽر *) tile1)[r1] = vh1;

            إذا (dst_مفرد_tiles) {
                HVX_ويڪٽر vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_ويڪٽر vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_مفرد_tiles + (r0 * n_کانو_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_مفرد_tiles + (r0 * n_کانو_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_ويڪٽر *) tile0_s)[r1] = vh0_s;
                ((HVX_ويڪٽر *) tile1_s)[r1] = vh1_s;
            }

            إذا (dst_f16) {
                HVX_ويڪٽر_Pair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_کانن_ڳڻپ + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_کانن_ڳڻپ + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ساكن مضمن فراغ hvx_transpose_32x32_لفظ(HVX_ويڪٽر * محدود m, HVX_ويڪٽر * محدود tmp) {
    لكل (صحيح i = 0; i < 16; ++i) {
        HVX_ويڪٽر_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    لكل (صحيح b = 0; b < 32; b += 4) {
        HVX_ويڪٽر_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ويڪٽر_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    لكل (صحيح b = 0; b < 32; b += 8) {
        لكل (صحيح i = 0; i < 4; ++i) {
            HVX_ويڪٽر_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    لكل (صحيح b = 0; b < 32; b += 16) {
        لكل (صحيح i = 0; i < 8; ++i) {
            HVX_ويڪٽر_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    لكل (صحيح i = 0; i < 16; ++i) {
        HVX_ويڪٽر_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    لكل (صحيح i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

ساكن مضمن فراغ gdn_pack_d_t_صف_tiles(
    __fp16 * محدود dst_tiles,
    ثابت __fp16 * محدود src_d,
    uint32_t S_v,
    HVX_ويڪٽر * محدود m,
    HVX_ويڪٽر * محدود tmp
) {
    لكل (uint32_t کانو_half = 0; کانو_half < S_v / 64; ++کانو_half) {
        uint32_t r0_بنياد = کانو_half * 2;
        لكل (uint32_t c0 = 0; c0 < 2; ++c0) {
            لكل (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + کانو_half * 64);
            }

            hvx_transpose_32x32_لفظ(m, tmp);

            uint32_t tile0_idx = (r0_بنياد + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_بنياد + 1) * 2 + c0;
            HVX_ويڪٽر * t0 = (HVX_ويڪٽر *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_ويڪٽر * t1 = (HVX_ويڪٽر *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            لكل (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

ساكن __attribute__((noinline)) فراغ gdn_build_inv_l_بلاڪون(
    __fp16 * محدود inv_صف_tiles,
    ثابت HVX_ويڪٽر * محدود صفون_kk,
    ثابت __fp16 * محدود decay_m,
    ثابت عائم * محدود beta,
    __fp16 * محدود l10_tile,
    __fp16 * محدود neg_a11_tile
) {
    ثابت HVX_ويڪٽر v_one_f16 = hvx_vec_splat_f16(1.0f);
    ثابت HVX_ويڪٽر_Pred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_ويڪٽر * محدود p_l00 = (HVX_ويڪٽر *) l00;
    HVX_ويڪٽر * محدود p_l11 = (HVX_ويڪٽر *) l11;
    HVX_ويڪٽر * محدود p_l10_tile = (HVX_ويڪٽر *) l10_tile;

    HVX_ويڪٽر * محدود tile00 = (HVX_ويڪٽر *) (inv_صف_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_ويڪٽر * محدود tile01 = (HVX_ويڪٽر *) (inv_صف_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_ويڪٽر * محدود tile11 = (HVX_ويڪٽر *) (inv_صف_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_ويڪٽر * محدود p_neg_a11 = (HVX_ويڪٽر *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_منزل_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    لكل (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    لكل (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_ويڪٽر v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_ويڪٽر v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_ويڪٽر v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_ويڪٽر v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_ويڪٽر r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفون_kk[t0], v_d0), v_b0);
        HVX_ويڪٽر r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفون_kk[t1], v_d1), v_b1);

        p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
    }

    لكل (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 32 + 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_ويڪٽر v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_ويڪٽر v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_ويڪٽر v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_ويڪٽر v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_ويڪٽر r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفون_kk[t0], v_d0), v_b0);
        HVX_ويڪٽر r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفون_kk[t1], v_d1), v_b1);

        HVX_ويڪٽر_Pair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
        p_l10_tile[r] = Q6_V_lo_W(vp_l10);
        p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
    }

    HVX_ويڪٽر a_صفون[32];
    لكل (uint32_t t = 0; t < 32; ++t) {
        HVX_ويڪٽر v_inv = Q6_V_vzero();
        لكل (uint32_t k = 0; k < t; ++k) {
            HVX_ويڪٽر v_lk = Q6_Vh_vsplat_R(l00[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_صفون[k]));
        }
        HVX_ويڪٽر_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_صفون[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    لكل (uint32_t r = 0; r < 16; ++r) {
        HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(a_صفون[2 * r + 1], a_صفون[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    لكل (uint32_t t = 0; t < 32; ++t) {
        HVX_ويڪٽر v_inv = Q6_V_vzero();
        لكل (uint32_t k = 0; k < t; ++k) {
            HVX_ويڪٽر v_lk = Q6_Vh_vsplat_R(l11[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_صفون[k]));
        }
        HVX_ويڪٽر_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_صفون[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    لكل (uint32_t r = 0; r < 16; ++r) {
        HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(a_صفون[2 * r + 1], a_صفون[2 * r + 0], -2);
        tile11[r] = Q6_V_lo_W(vp);

        HVX_ويڪٽر n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_صفون[2 * r + 0]);
        HVX_ويڪٽر n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_صفون[2 * r + 1]);
        HVX_ويڪٽر_Pair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
        p_neg_a11[r] = Q6_V_lo_W(vp_neg);
    }
}


ساكن مضمن فراغ gdn_dma_ڌڪو_chunk_inputs(
    dma_قطار * dma_q,
    عائم * vtcm_q,
    عائم * vtcm_k,
    عائم * vtcm_v,
    ثابت بنية htp_tensor * q,
    ثابت بنية htp_tensor * k,
    ثابت بنية htp_tensor * v,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_سائيز,
    uint32_t S_v
) {
    ثابت dma_addr_t q_dma = q->معلومات + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
    ثابت dma_addr_t k_dma = k->معلومات + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
    ثابت dma_addr_t v_dma = v->معلومات + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_q, q_dma), S_v * حجم(عائم), q->nb[2], S_v * حجم(عائم), chunk_سائيز);
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_k, k_dma), S_v * حجم(عائم), k->nb[2], S_v * حجم(عائم), chunk_سائيز);
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_v, v_dma), S_v * حجم(عائم), v->nb[2], S_v * حجم(عائم), chunk_سائيز);
}

ساكن مضمن فراغ gdn_dma_ڌڪو_chunk_gb(
    dma_قطار * dma_q,
    عائم * vtcm_g_raw,
    عائم * vtcm_b_raw,
    ثابت بنية htp_tensor * g,
    ثابت بنية htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_سائيز,
    uint32_t n_batch
) {
    ثابت dma_addr_t g_dma    = g->معلومات + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    ثابت dma_addr_t beta_dma = beta->معلومات + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    ثابت uint32_t صف_بائيٽون  = n_batch * حجم(عائم);

    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_g_raw, g_dma), صف_بائيٽون, g->nb[2], صف_بائيٽون, chunk_سائيز);
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_b_raw, beta_dma), صف_بائيٽون, beta->nb[2], صف_بائيٽون, chunk_سائيز);
}

ساكن مضمن فراغ gdn_pack_s_کانو_tiles(
    __fp16 * محدود vtcm_s_کانو_tiles,
    __fp16 * محدود vtcm_s_f16,
    ثابت عائم * محدود vtcm_s_state,
    uint32_t S_v
) {
    لكل (uint32_t j = 0; j < S_v; ++j) {
        لكل (uint32_t i = 0; i < S_v; i += 64) {
            HVX_ويڪٽر v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_ويڪٽر v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_منزل_f16(v0, v1);
        }
    }
    hmx_interleave_صفون_منزل_tiles(vtcm_s_کانو_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);
}

بنية htp_gdn_سر_ptrs {
    عائم *  s_state;
    __fp16 * s_f16;
    __fp16 * s_کانو_tiles;
    عائم *  s_update_f32;
    __fp16 * s_update_tiles;

    عائم * q_f32[2];
    عائم * k_f32[2];
    عائم * v_f32[2];
    عائم * g_f32[2];
    عائم * b_f32[2];
    عائم * o_f32[2];

    عائم * v_inter_f32;
    عائم * o_inter_f32;
    عائم * o_intra_f32;

    __fp16 * k_f16;
    __fp16 * v_مفرد_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_صف_tiles;
    __fp16 * q_مفرد_صف_tiles;
    __fp16 * k_صف_tiles;
    __fp16 * k_کانو_tiles;
    __fp16 * k_مفرد_صف_tiles;
    __fp16 * k_کانو_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_صف_tiles;
    __fp16 * a_صف_tiles;
    __fp16 * v_مفرد_کانو_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_کانو_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_صف_tiles;

    __fp16 * gamma;
    عائم *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_ويڪٽر * صفون_kk;
    HVX_ويڪٽر * صفون_qk;
    HVX_ويڪٽر * صفون_inv;
    HVX_ويڪٽر * صفون_a;

    HVX_ويڪٽر * vtcm_m;
    HVX_ويڪٽر * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_اندريون_dma;
    dma_addr_t state_ٻاهريون_dma;
};

ساكن مضمن فراغ gdn_init_سر_ptrs(
    بنية htp_gdn_سر_ptrs * سر,
    ثابت بنية htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_بنياد,
    uint32_t h,
    uint32_t بنياد_iv1,
    uint32_t iv3,
    ثابت بنية htp_tensor * q,
    ثابت بنية htp_tensor * k,
    ثابت بنية htp_tensor * v,
    ثابت بنية htp_tensor * state,
    ثابت بنية htp_tensor * dst,
    ثابت بنية htp_tensor * dst_cache,
    ثابت بنية htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_ٽوڪن,
    uint32_t chunk_سائيز
) {
    ثابت size_t dma_scalar_sz = hex_ڦيرو_مٿي(chunk_سائيز * حجم(عائم), 128);
    ثابت size_t decay_sz      = 64 * 64 * حجم(__fp16);
    ثابت size_t صف_vecs_sz   = 64 * 128;

    سر->s_state        = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_s_state + h * L->state_f32_بائيٽون);
    سر->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_s_f16 + h * L->state_f16_بائيٽون);
    سر->s_کانو_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_s_کانو_tiles + h * L->state_tiles_بائيٽون);
    سر->s_update_f32   = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_s_update_f32 + h * L->state_f32_بائيٽون);
    سر->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_s_update_tiles + h * L->state_tiles_بائيٽون);

    سر->q_f32[0] = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_q_f32[0] + h * L->dma_chunk_بائيٽون);
    سر->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_q_f32[1] + h * L->dma_chunk_بائيٽون) : سر->q_f32[0];
    سر->k_f32[0] = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_k_f32[0] + h * L->dma_chunk_بائيٽون);
    سر->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_k_f32[1] + h * L->dma_chunk_بائيٽون) : سر->k_f32[0];
    سر->v_f32[0] = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_v_f32[0] + h * L->dma_chunk_بائيٽون);
    سر->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_v_f32[1] + h * L->dma_chunk_بائيٽون) : سر->v_f32[0];
    سر->g_f32[0] = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_g_f32[0] + h * dma_scalar_sz);
    سر->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_g_f32[1] + h * dma_scalar_sz) : سر->g_f32[0];
    سر->b_f32[0] = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_b_f32[0] + h * dma_scalar_sz);
    سر->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_b_f32[1] + h * dma_scalar_sz) : سر->b_f32[0];
    سر->o_f32[0] = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_o_f32[0] + h * L->dma_chunk_بائيٽون);
    سر->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_o_f32[1] + h * L->dma_chunk_بائيٽون) : سر->o_f32[0];

    سر->v_inter_f32 = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_v_inter_f32 + h * L->dma_chunk_بائيٽون);
    سر->o_inter_f32 = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_o_inter_f32 + h * L->dma_chunk_بائيٽون);
    سر->o_intra_f32 = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_o_intra_f32 + h * L->dma_chunk_بائيٽون);

    سر->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_k_f16 + h * L->act_f16_بائيٽون);
    سر->v_مفرد_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_v_مفرد_f16 + h * L->act_f16_بائيٽون);
    سر->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_delta_f16 + h * L->act_f16_بائيٽون);
    سر->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_d_f16 + h * L->act_f16_بائيٽون);

    سر->q_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_q_صف_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->q_مفرد_صف_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_q_مفرد_صف_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->k_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_k_صف_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->k_کانو_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_k_کانو_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->k_مفرد_صف_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_k_مفرد_صف_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->k_کانو_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_k_کانو_tiles_64x128 + h * L->tile_64x_Sv_بائيٽون);
    سر->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_kk_tiles + h * L->tile_64x64_بائيٽون);
    سر->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_qk_tiles + h * L->tile_64x64_بائيٽون);
    سر->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_v_inter_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_o_inter_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->inv_صف_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_inv_صف_tiles + h * L->tile_64x64_بائيٽون);
    سر->a_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_a_صف_tiles + h * L->tile_64x64_بائيٽون);
    سر->v_مفرد_کانو_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_v_مفرد_کانو_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_delta_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->delta_کانو_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_delta_کانو_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_o_intra_tiles + h * L->tile_64x_Sv_بائيٽون);
    سر->d_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_d_صف_tiles + h * L->tile_64x_Sv_بائيٽون);

    سر->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_gamma + h * dma_scalar_sz);
    سر->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_gamma + h * dma_scalar_sz + 128);
    سر->lambda_init     = VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L->off_lambda_init + h * dma_scalar_sz);
    سر->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_decay_m + h * decay_sz);
    سر->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_بنياد, L->off_decay_a + h * decay_sz);

    سر->صفون_kk  = VTCM_LAYOUT_PTR(HVX_ويڪٽر, vtcm_بنياد, L->off_صفون_kk + h * صف_vecs_sz);
    سر->صفون_qk  = VTCM_LAYOUT_PTR(HVX_ويڪٽر, vtcm_بنياد, L->off_صفون_qk + h * صف_vecs_sz);
    سر->صفون_inv = VTCM_LAYOUT_PTR(HVX_ويڪٽر, vtcm_بنياد, L->off_صفون_inv + h * صف_vecs_sz);
    سر->صفون_a   = VTCM_LAYOUT_PTR(HVX_ويڪٽر, vtcm_بنياد, L->off_صفون_a + h * صف_vecs_sz);

    سر->vtcm_m   = VTCM_LAYOUT_PTR(HVX_ويڪٽر, vtcm_بنياد, L->off_thread_scratch + h * (64 * 128));
    سر->vtcm_tmp = سر->vtcm_m + 32;

    سر->iv1 = بنياد_iv1 + h;
    سر->iv3 = iv3;
    سر->iq1 = fastmodulo(سر->iv1, q->ne[1], &kparams->div_q1);
    سر->ik1 = fastmodulo(سر->iv1, k->ne[1], &kparams->div_k1);
    سر->iq3 = fastdiv(سر->iv3, &kparams->div_rq3);
    سر->ik3 = fastdiv(سر->iv3, &kparams->div_rk3);

    سر->state_اندريون_dma = state->معلومات +
        ((uint64_t) سر->iv3 * kparams->state_seq_stride + (uint64_t) سر->iv1 * S_v * S_v) * حجم(عائم);

    سر->state_ٻاهريون_dma = dst_cache ?
        (dst_cache->معلومات + ((uint64_t) سر->iv3 * H + سر->iv1) * S_v * S_v * حجم(عائم)) :
        (dst->معلومات + ((uint64_t) S_v * H * n_ٽوڪن * kparams->n_seqs + (uint64_t) (سر->iv3 * H + سر->iv1) * S_v * S_v) * حجم(عائم));
}

بنية htp_gdn_batch_context {
    بنية htp_gdn_سر_ptrs * heads;
    ثابت عائم *              vtcm_g_raw;
    ثابت عائم *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    عائم                      scale;
    بنية htp_ops_context *   octx;
    ثابت بنية htp_gdn_kernel_params * kparams;
};

ساكن فراغ gdn_hvx_init_state_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    gdn_pack_s_کانو_tiles(سر->s_کانو_tiles, سر->s_f16, سر->s_state, bctx->S_v);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

ساكن مضمن __attribute__((اڻ_استعمال_ٿيل_2)) HVX_ويڪٽر hvx_clamp_neg20_0(HVX_ويڪٽر v, HVX_ويڪٽر v_zero, HVX_ويڪٽر v_neg20) {
    HVX_ويڪٽر_Pred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_ويڪٽر_Pred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    أرجع Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

ساكن مضمن HVX_ويڪٽر hvx_اڳياڙي_جاچ_f32(HVX_ويڪٽر v, HVX_ويڪٽر حاصل_اندريون) {
    ثابت HVX_ويڪٽر zero = Q6_V_vzero();

    v = hvx_vec_جوڙيو_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_جوڙيو_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_جوڙيو_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_جوڙيو_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_جوڙيو_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_جوڙيو_f32_f32(v, حاصل_اندريون);

    أرجع v;
}

ساكن مضمن HVX_ويڪٽر hvx_splat_آخري_f32(HVX_ويڪٽر v) {
    أرجع hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساكن فراغ gdn_hvx_phase1a_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;
    ثابت uint32_t n_batch = bctx->n_batch;

    إذا (n_batch == 1) {
        hvx_vmem(سر->g_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(سر->g_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(سر->b_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(سر->b_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } وإلا {
        int32_t offsets[32] __attribute__((aligned(128)));
        لكل (صحيح k = 0; k < 32; ++k) {
            offsets[k] = k * n_batch * حجم(عائم);
        }
        HVX_ويڪٽر vv = *(ثابت HVX_ويڪٽر *) offsets;
        ثابت size_t rt_g = (size_t) ((ثابت uint8_t *) bctx->vtcm_g_raw + i * حجم(عائم));
        ثابت size_t rt_b = (size_t) ((ثابت uint8_t *) bctx->vtcm_b_raw + i * حجم(عائم));
        ثابت size_t mu   = 64 * n_batch * حجم(عائم);

        Q6_vgather_ARMVw((HVX_ويڪٽر *) (سر->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_ويڪٽر *) (سر->g_f32[curr_buf] + 32), rt_g + 32 * n_batch * حجم(عائم), mu, vv);
        Q6_vgather_ARMVw((HVX_ويڪٽر *) (سر->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_ويڪٽر *) (سر->b_f32[curr_buf] + 32), rt_b + 32 * n_batch * حجم(عائم), mu, vv);
    }

    ثابت uint32_t t_chunk = bctx->c * 64;
    ثابت uint32_t valid_ٽوڪن = hex_smin(64, bctx->kparams->n_ٽوڪن - t_chunk);
    إذا (valid_ٽوڪن < 64) {
        لكل (uint32_t t = valid_ٽوڪن; t < 64; ++t) {
            سر->g_f32[curr_buf][t] = 0.0f;
            سر->b_f32[curr_buf][t] = 0.0f;
        }
        ثابت HVX_ويڪٽر vzero = Q6_V_vzero();
        لكل (uint32_t t = valid_ٽوڪن; t < 64; ++t) {
            لكل (uint32_t j = 0; j < S_v; j += 32) {
                hvx_vmem(سر->q_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(سر->k_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(سر->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    ثابت HVX_ويڪٽر v_g0 = hvx_vmem(سر->g_f32[curr_buf] + 0);
    ثابت HVX_ويڪٽر v_g1 = hvx_vmem(سر->g_f32[curr_buf] + 32);

    HVX_ويڪٽر v_gamma0 = hvx_اڳياڙي_جاچ_f32(v_g0, Q6_V_vzero());
    HVX_ويڪٽر v_حاصل  = hvx_splat_آخري_f32(v_gamma0);
    HVX_ويڪٽر v_gamma1 = hvx_اڳياڙي_جاچ_f32(v_g1, v_حاصل);

    ثابت HVX_ويڪٽر v_zero  = Q6_V_vzero();
    ثابت HVX_ويڪٽر v_neg20 = hvx_vec_splat_f32(-20.0f);

    hvx_vmem(سر->gamma) = hvx_vec_f32_منزل_f16(v_gamma0, v_gamma1);

    HVX_ويڪٽر v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_ويڪٽر v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    hvx_vmem(سر->lambda_init + 0)  = v_l0;
    hvx_vmem(سر->lambda_init + 32) = v_l1;
    hvx_vmem(سر->lambda_init_f16)  = hvx_vec_f32_منزل_f16(v_l0, v_l1);

    gdn_f32_منزل_hmx_صف_tiles_and_f16(سر->k_صف_tiles, سر->k_مفرد_صف_tiles, سر->k_f16,
                                     سر->k_f32[curr_buf], سر->lambda_init_f16, 64, S_v);
    hmx_interleave_صفون_منزل_tiles(سر->k_کانو_tiles, سر->k_f16, 64, S_v, S_v, 0, 64);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ساكن فراغ gdn_hvx_phase1b_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;

    gdn_f32_منزل_hmx_صف_tiles_and_f16(سر->q_صف_tiles, سر->q_مفرد_صف_tiles, عدم,
                                     سر->q_f32[curr_buf], سر->lambda_init_f16, 64, S_v);

    hmx_interleave_کانن_ڳڻپ_منزل_tiles(سر->k_کانو_tiles_64x128, سر->k_f16, 64, S_v, S_v, 2, 0, 64);

    ثابت uint16_t * gamma_u16 = (ثابت uint16_t *) سر->gamma;
    ثابت HVX_ويڪٽر v_gamma   = hvx_vmem(سر->gamma);

    ثابت HVX_ويڪٽر v_zero_f16  = Q6_V_vzero();
    ثابت HVX_ويڪٽر v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    ثابت HVX_ويڪٽر v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    ثابت HVX_ويڪٽر v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(سر->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(سر->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    لكل (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_ويڪٽر v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_ويڪٽر v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_ويڪٽر diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_ويڪٽر diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_ويڪٽر_Pred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_ويڪٽر_Pred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_ويڪٽر diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_ويڪٽر diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_ويڪٽر v_exp0      = hvx_vec_exp2_f16(diff_log2e0);
        HVX_ويڪٽر v_exp1      = hvx_vec_exp2_f16(diff_log2e1);

        HVX_ويڪٽر_Pred نقاب_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_ويڪٽر_Pred نقاب_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_ويڪٽر v_m0         = Q6_V_vand_QV(نقاب_lt0, v_exp0);
        HVX_ويڪٽر v_m1         = Q6_V_vand_QV(نقاب_lt1, v_exp1);

        HVX_ويڪٽر_Pred نقاب_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_ويڪٽر_Pred نقاب_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_ويڪٽر_Pred نقاب_diag0 = Q6_Q_and_QQn(نقاب_le0, نقاب_lt0);
        HVX_ويڪٽر_Pred نقاب_diag1 = Q6_Q_and_QQn(نقاب_le1, نقاب_lt1);

        HVX_ويڪٽر v_a0         = Q6_V_vmux_QVV(نقاب_diag0, v_one_f16, v_m0);
        HVX_ويڪٽر v_a1         = Q6_V_vmux_QVV(نقاب_diag1, v_one_f16, v_m1);

        hvx_vmem(سر->decay_m + t0 * 64) = v_m0;
        hvx_vmem(سر->decay_a + t0 * 64) = v_a0;
        hvx_vmem(سر->decay_m + t1 * 64) = v_m1;
        hvx_vmem(سر->decay_a + t1 * 64) = v_a1;
    }

    {
        HVX_ويڪٽر v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_ويڪٽر diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_ويڪٽر_Pred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_ويڪٽر diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_ويڪٽر v_exp         = hvx_vec_exp2_f16(diff_log2e);

        HVX_ويڪٽر_Pred نقاب_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_ويڪٽر v_m           = Q6_V_vand_QV(نقاب_lt_t, v_exp);

        HVX_ويڪٽر_Pred نقاب_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_ويڪٽر_Pred نقاب_diag = Q6_Q_and_QQn(نقاب_le_t, نقاب_lt_t);
        HVX_ويڪٽر v_a           = Q6_V_vmux_QVV(نقاب_diag, v_one_f16, v_m);

        hvx_vmem(سر->decay_m + 63 * 64) = v_m;
        hvx_vmem(سر->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ساكن فراغ gdn_hvx_phase2_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;

    gdn_unpack_64x64_tiles_منزل_vectors(سر->صفون_kk, سر->kk_tiles);

    gdn_build_inv_l_بلاڪون(
        سر->inv_صف_tiles,
        سر->صفون_kk,
        سر->decay_m,
        سر->b_f32[curr_buf],
        (__fp16 *) سر->vtcm_m,
        (__fp16 *) سر->vtcm_m + HMX_FP16_TILE_N_ELMS
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

ساكن فراغ gdn_hvx_phase3_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_منزل_f32(سر->v_inter_f32, سر->v_inter_tiles, S_v);

    HVX_ويڪٽر_Alias local_b[2];
    local_b[0].v = hvx_vmem(سر->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(سر->b_f32[curr_buf] + 32);

    لكل (uint32_t t = 0; t < 64; ++t) {
        HVX_ويڪٽر vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
        لكل (uint32_t j = 0; j < S_v; j += 64) {
            HVX_ويڪٽر vv0 = hvx_vmem(سر->v_f32[curr_buf] + t * S_v + j + 0);
            HVX_ويڪٽر vv1 = (j + 32 < S_v) ? hvx_vmem(سر->v_f32[curr_buf] + t * S_v + j + 32) : Q6_V_vzero();
            HVX_ويڪٽر vi0 = hvx_vmem(سر->v_inter_f32 + t * S_v + j + 0);
            HVX_ويڪٽر vi1 = (j + 32 < S_v) ? hvx_vmem(سر->v_inter_f32 + t * S_v + j + 32) : Q6_V_vzero();

            HVX_ويڪٽر vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
            HVX_ويڪٽر vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

            hvx_vmem(سر->v_مفرد_f16 + t * S_v + j) = hvx_vec_f32_منزل_f16(vp0, vp1);
        }
    }

    hmx_interleave_کانن_ڳڻپ_منزل_tiles(سر->v_مفرد_کانو_tiles, سر->v_مفرد_f16, 64, S_v, S_v, 2, 0, 64);

    gdn_unpack_64x64_tiles_منزل_vectors(سر->صفون_qk, سر->qk_tiles);
    لكل (uint32_t t = 0; t < 64; ++t) {
        HVX_ويڪٽر v_decay_a = hvx_vmem(سر->decay_a + t * 64);
        سر->صفون_a[t]      = hvx_vec_mul_f16_f16(سر->صفون_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_منزل_tiles(سر->a_صف_tiles, سر->صفون_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

ساكن فراغ gdn_hvx_phase4_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_منزل_f16(سر->delta_f16, سر->delta_tiles, S_v);
    hmx_interleave_کانن_ڳڻپ_منزل_tiles(سر->delta_کانو_tiles, سر->delta_f16, 64, S_v, S_v, 2, 0, 64);

    ثابت uint16_t * decay_آخري = (ثابت uint16_t *) (سر->decay_a + 63 * 64);
    ثابت HVX_ويڪٽر vzero      = Q6_V_vzero();

    لكل (uint32_t s = 0; s < 64; ++s) {
        HVX_ويڪٽر vs         = Q6_Vh_vsplat_R(decay_آخري[s]);
        HVX_ويڪٽر_Pred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
        لكل (uint32_t j = 0; j < S_v; j += 64) {
            HVX_ويڪٽر vd   = hvx_vmem(سر->delta_f16 + s * S_v + j);
            HVX_ويڪٽر prod = hvx_vec_mul_f16_f16(vd, vs);
            hvx_vmem(سر->d_f16 + s * S_v + j) = Q6_V_vmux_QVV(p_zero, vzero, prod);
        }
    }

    gdn_pack_d_t_صف_tiles(سر->d_صف_tiles, سر->d_f16, S_v, سر->vtcm_m, سر->vtcm_tmp);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

ساكن فراغ gdn_hvx_phase5_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_ٻاهريون, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;
    ثابت عائم scale = bctx->scale;

    gdn_unpack_64x_S_tiles_منزل_f32(سر->o_inter_f32, سر->o_inter_tiles, S_v);
    gdn_unpack_64x_S_tiles_منزل_f32(سر->o_intra_f32, سر->o_intra_tiles, S_v);

    HVX_ويڪٽر vscale = hvx_vec_splat_f32(scale);
    لكل (uint32_t j = 0; j < 64 * S_v / 32; ++j) {
        HVX_ويڪٽر vi = hvx_vmem(سر->o_inter_f32 + j * 32);
        HVX_ويڪٽر va = hvx_vmem(سر->o_intra_f32 + j * 32);
        hvx_vmem(سر->o_f32[curr_buf] + j * 32) = hvx_vec_mul_f32_f32(hvx_vec_جوڙيو_f32_f32(vi, va), vscale);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_ٻاهريون, info);
}

ساكن فراغ gdn_hvx_phase6_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    (فراغ) n;
    بنية htp_gdn_batch_context * bctx = (بنية htp_gdn_batch_context *) معلومات;
    بنية htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    بنية htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t S_v = bctx->S_v;
    ثابت uint32_t c = bctx->c;
    ثابت uint32_t n_chunks = bctx->kparams->n_chunks;

    gdn_unpack_Sx_S_tiles_منزل_f32(سر->s_update_f32, سر->s_update_tiles, S_v);

    HVX_ويڪٽر_Alias آخري_lambda;
    آخري_lambda.v = hvx_vmem(سر->lambda_init + 32);
    HVX_ويڪٽر v_l_final = hvx_vec_splat_f32(آخري_lambda.fp32[31]);

    لكل (uint32_t j = 0; j < S_v * S_v / 32; ++j) {
        HVX_ويڪٽر vs_old = hvx_vmem(سر->s_state + j * 32);
        HVX_ويڪٽر vsu    = hvx_vmem(سر->s_update_f32 + j * 32);
        hvx_vmem(سر->s_state + j * 32) = hvx_vec_جوڙيو_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
    }

    إذا (c + 1 < n_chunks) {
        gdn_pack_s_کانو_tiles(سر->s_کانو_tiles, سر->s_f16, سر->s_state, S_v);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


ساكن صحيح gated_delta_net_f32_hmx_chunked(
    بنية htp_ops_context * octx,
    ثابت بنية htp_gdn_kernel_params * kparams,
    uint32_t صف_شروعات,
    uint32_t nrows
) {
    ثابت بنية htp_tensor * q         = octx->src[0];
    ثابت بنية htp_tensor * k         = octx->src[1];
    ثابت بنية htp_tensor * v         = octx->src[2];
    ثابت بنية htp_tensor * g         = octx->src[3];
    ثابت بنية htp_tensor * beta      = octx->src[4];
    ثابت بنية htp_tensor * state     = octx->src[5];
    ثابت بنية htp_tensor * dst       = octx->dst;
    ثابت بنية htp_tensor * dst_cache = octx->dsts[1];

    ثابت uint32_t S_v        = kparams->S_v;
    ثابت uint32_t H          = kparams->H;
    ثابت uint32_t n_ٽوڪن   = kparams->n_ٽوڪن;
    ثابت عائم    scale      = kparams->scale;
    ثابت uint32_t chunk_سائيز = kparams->chunk_سائيز;
    ثابت uint32_t n_chunks   = kparams->n_chunks;
    ثابت uint32_t n_sv_tiles = S_v / 32;

    بنية htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_سائيز, kparams->n_heads_batch, kparams->n_ڌاڳا, kparams->pipeline != 0);

    إذا (L.ڪل_بائيٽون > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت vtcm_بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;

    عائم * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L.off_g_raw[0])
    };
    عائم * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(عائم, vtcm_بنياد, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_بنياد, L.off_scales_1);
    hmx_init_ڪالم_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_قطار_t  hmx_q = octx->ctx->hmx_قطار;
    dma_قطار *  dma_q = octx->ctx->dma[0];
    work_قطار_t wp    = octx->ctx->work_قطار;

    بنية htp_gdn_سر_ptrs heads[8];
    بنية htp_gdn_hmx_gemm_task gemm_tasks[8][9];

    uint32_t n_batch = 1;
    لكل (uint32_t r = صف_شروعات; r < صف_شروعات + nrows; r += n_batch) {
        ثابت uint32_t سر_اندريون_seq         = fastmodulo(r, H, &kparams->div_H);
        ثابت uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        ثابت uint32_t heads_کاٻو_اندريون_seq   = H - سر_اندريون_seq;
        ثابت uint32_t heads_کاٻو_اندريون_range = (صف_شروعات + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_کاٻو_اندريون_seq, heads_کاٻو_اندريون_range));

        لكل (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_سر_ptrs(&heads[h], &L, vtcm_بنياد, h, سر_اندريون_seq, iv3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_ٽوڪن, chunk_سائيز);
        }

        بنية htp_gdn_batch_context bctx;
        bctx.heads      = heads;
        bctx.vtcm_g_raw = عدم;
        bctx.vtcm_b_raw = عدم;
        bctx.curr_buf   = 0;
        bctx.c          = 0;
        bctx.n_batch    = n_batch;
        bctx.S_v        = S_v;
        bctx.scale      = scale;
        bctx.octx       = octx;
        bctx.kparams    = kparams;

        لكل (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(heads[h].s_state, heads[h].state_اندريون_dma),
                           S_v * حجم(عائم), S_v * حجم(عائم), S_v * حجم(عائم), S_v);
        }
        لكل (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_ڪڍو(dma_q);
        }

        إذا (n_chunks > 0) {
            work_قطار_لاڳيتو(wp, gdn_hvx_init_state_ڪم_ڪندڙ, &bctx, n_batch);

            ثابت uint32_t chunk0_ٽوڪن = hex_smin(chunk_سائيز, n_ٽوڪن);
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                gdn_dma_ڌڪو_chunk_inputs(dma_q, heads[h].q_f32[0], heads[h].k_f32[0], heads[h].v_f32[0],
                                          q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                          heads[h].iv3, heads[h].iv1, 0, chunk0_ٽوڪن, S_v);
            }
            gdn_dma_ڌڪو_chunk_gb(dma_q, vtcm_g_raw[0], vtcm_b_raw[0], g, beta, iv3, سر_اندريون_seq, 0, chunk0_ٽوڪن, n_batch);
        }

        لكل (uint32_t c = 0; c < n_chunks; ++c) {
            ثابت uint32_t curr_buf = c & 1;
            ثابت uint32_t ايندڙ_buf = (c + 1) & 1;
            ثابت uint32_t t_chunk  = c * chunk_سائيز;

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                dma_قطار_ڪڍو(dma_q);
                dma_قطار_ڪڍو(dma_q);
                dma_قطار_ڪڍو(dma_q);
            }
            dma_قطار_ڪڍو(dma_q);
            dma_قطار_ڪڍو(dma_q);

            إذا (c + 1 < n_chunks) {
                ثابت uint32_t ايندڙ_t_chunk = (c + 1) * chunk_سائيز;
                ثابت uint32_t ايندڙ_ٽوڪن_2  = hex_smin(chunk_سائيز, n_ٽوڪن - ايندڙ_t_chunk);
                لكل (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_ڌڪو_chunk_inputs(dma_q, heads[h].q_f32[ايندڙ_buf], heads[h].k_f32[ايندڙ_buf], heads[h].v_f32[ايندڙ_buf],
                                              q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                              heads[h].iv3, heads[h].iv1, ايندڙ_t_chunk, ايندڙ_ٽوڪن_2, S_v);
                }
                gdn_dma_ڌڪو_chunk_gb(dma_q, vtcm_g_raw[ايندڙ_buf], vtcm_b_raw[ايندڙ_buf],
                                      g, beta, iv3, سر_اندريون_seq, ايندڙ_t_chunk, ايندڙ_ٽوڪن_2, n_batch);
            }

            إذا (c > 0) {
                لكل (uint32_t h = 0; h < n_batch; ++h) {
                    dma_قطار_ڪڍو(dma_q);
                }
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase1a_ڪم_ڪندڙ, &bctx, n_batch);

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][0], heads[h].k_صف_tiles, heads[h].k_کانو_tiles, heads[h].kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][2], heads[h].k_مفرد_صف_tiles, heads[h].s_کانو_tiles, heads[h].v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase1b_ڪم_ڪندڙ, &bctx, n_batch);

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][1], heads[h].q_صف_tiles, heads[h].k_کانو_tiles, heads[h].qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][3], heads[h].q_مفرد_صف_tiles, heads[h].s_کانو_tiles, heads[h].o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase2_ڪم_ڪندڙ, &bctx, n_batch);

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][7],
                    (__fp16 *) heads[h].vtcm_m,
                    heads[h].inv_صف_tiles + 0 * HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    1, 1, 1, vtcm_scales_1
                );
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][8],
                    (__fp16 *) heads[h].vtcm_m + HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    heads[h].inv_صف_tiles + 2 * HMX_FP16_TILE_N_ELMS,
                    1, 1, 1, vtcm_scales_1
                );
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase3_ڪم_ڪندڙ, &bctx, n_batch);

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][4], heads[h].inv_صف_tiles, heads[h].v_مفرد_کانو_tiles, heads[h].delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase4_ڪم_ڪندڙ, &bctx, n_batch);

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][5], heads[h].a_صف_tiles, heads[h].delta_کانو_tiles, heads[h].o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڌڪو_hmx_gemm_task(hmx_q, &gemm_tasks[h][6], heads[h].d_صف_tiles, heads[h].k_کانو_tiles_64x128, heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);
            }

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase5_ڪم_ڪندڙ, &bctx, n_batch);

            ثابت uint32_t valid_ٽوڪن = hex_smin(chunk_سائيز, n_ٽوڪن - t_chunk);
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                ثابت dma_addr_t attn_chunk_dma = dst->معلومات +
                    ((uint64_t) heads[h].iv3 * n_ٽوڪن * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * حجم(عائم);
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * حجم(عائم), S_v * حجم(عائم), valid_ٽوڪن);
            }

            لكل (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_ڪڍو(hmx_q);
            }

            work_قطار_لاڳيتو(wp, gdn_hvx_phase6_ڪم_ڪندڙ, &bctx, n_batch);
        }

        إذا (n_chunks > 0) {
            لكل (uint32_t h = 0; h < n_batch; ++h) {
                dma_قطار_ڪڍو(dma_q);
            }
        }

        لكل (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(heads[h].state_ٻاهريون_dma, heads[h].s_state),
                           S_v * حجم(عائم), S_v * حجم(عائم), S_v * حجم(عائم), S_v);
        }
        لكل (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_ڪڍو(dma_q);
        }
    }

    dma_قطار_flush(dma_q);
    أرجع HTP_حالت_OK;
}

صحيح op_gated_delta_net(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * q     = octx->src[0];
    ثابت بنية htp_tensor * k     = octx->src[1];
    ثابت بنية htp_tensor * v     = octx->src[2];
    ثابت بنية htp_tensor * g     = octx->src[3];
    ثابت بنية htp_tensor * beta  = octx->src[4];
    ثابت بنية htp_tensor * state = octx->src[5];
    ثابت بنية htp_tensor * dst   = octx->dst;

    إذا (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t S_v      = v->ne[0];
    ثابت uint32_t H        = v->ne[1];
    ثابت uint32_t n_ٽوڪن = v->ne[2];
    ثابت uint32_t n_seqs   = v->ne[3];
    ثابت uint32_t K        = octx->op_params[0];

    إذا (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_ٽوڪن == 0 || n_seqs == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    إذا ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    إذا (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_ٽوڪن || k->ne[2] != n_ٽوڪن || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    إذا (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    إذا (dst->ne[0] != S_v * H || dst->ne[1] != n_ٽوڪن * n_seqs + S_v * n_seqs * K) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    لكل (صحيح i = 0; i < 5; i++) {
        إذا (htp_tensor_is_extended(octx->src[i])) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
    }
    إذا (htp_tensor_is_extended(octx->dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    إذا (octx->dsts[1]) {
        ثابت بنية htp_tensor * dst_cache = octx->dsts[1];
        إذا (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
    }

    ثابت بنية htp_gdn_kernel_params * kparams = (ثابت بنية htp_gdn_kernel_params *) octx->kernel_params;
    بنية htp_gdn_kernel_params kparams_local;
    إذا (!kparams || kparams->S_v == 0) {
        ثابت uint32_t rq3 = n_seqs / q->ne[3];
        ثابت uint32_t rk3 = n_seqs / k->ne[3];
        ثابت uint32_t ڪل_صفون = H * n_seqs;
        uint32_t n_ڌاڳا = (ڪل_صفون < octx->n_ڌاڳا) ? ڪل_صفون : octx->n_ڌاڳا;
        إذا (n_ڌاڳا == 0) {
            n_ڌاڳا = 1;
        }

        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_ڌاڳا           = n_ڌاڳا;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_ٽوڪن            = n_ٽوڪن;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.ڪل_صفون          = ڪل_صفون;
        kparams_local.صفون_في_thread     = (ڪل_صفون + n_ڌاڳا - 1) / n_ڌاڳا;
        ثابت bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_ٽوڪن >= HTP_GDN_MIN_ٽوڪن) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        بنية htp_gdn_hmx_vtcm_layout hmx_layout_local;
        بنية htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        إذا (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_سائيز, ڪل_صفون, octx->ctx->vtcm_سائيز, n_ڌاڳا, true, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_سائيز      = HTP_GDN_CHUNK_سائيز;
            kparams_local.n_chunks        = (n_ٽوڪن + HTP_GDN_CHUNK_سائيز - 1) / HTP_GDN_CHUNK_سائيز;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_سائيز       = (uint32_t) hmx_layout_local.ڪل_بائيٽون;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_بائيٽون;
            kparams_local.vtcm_في_thread = (uint32_t) (hmx_layout_local.ڪل_بائيٽون / (n_ڌاڳا > 0 ? n_ڌاڳا : 1));
        } وإلا {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_ڌاڳا);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_في_thread = (uint32_t) hvx_layout_local.بائيٽون_في_thread;
            kparams_local.vtcm_سائيز       = (uint32_t) hvx_layout_local.ڪل_بائيٽون;
        }
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((عائم) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / حجم(عائم));
        kparams_local.state_سائيز_في_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_قدرون(H);
        kparams_local.div_q1        = init_fastdiv_قدرون(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_قدرون(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_قدرون(rq3);
        kparams_local.div_rk3       = init_fastdiv_قدرون(rk3);
        kparams_local.div_n_ڌاڳا = init_fastdiv_قدرون(n_ڌاڳا);

        kparams = &kparams_local;
    }

    ثابت uint32_t ڪل_صفون = kparams->ڪل_صفون;
    uint32_t صف_شروعات = 0;
    uint32_t nrows     = ڪل_صفون;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت bool can_ورهايو = htp_tensor_mdev_معلومات_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            ڪل_صفون,
            can_ورهايو ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.ڳڻپ,
            &octx->ctx->mdev.ڳڻپ_div
        );
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    } وإلا إذا (octx->op_params[1] != 0) {
        صف_شروعات = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        أرجع gated_delta_net_f32_hmx_chunked(octx, kparams, صف_شروعات, nrows);
    }

    ثابت uint32_t n_ڌاڳا = (nrows < kparams->n_ڌاڳا) ? nrows : kparams->n_ڌاڳا;

    بنية htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.صف_شروعات = صف_شروعات;
    gctx.nrows     = nrows;
    gctx.vtcm_بنياد = octx->ctx->vtcm_بنياد;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_ڌاڳا);

    إذا (gctx.layout.ڪل_بائيٽون > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    FARF(مٿاهون_2, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.ڪل_بائيٽون, n_ڌاڳا);

    إذا (n_ٽوڪن == 1) {
        work_قطار_لاڳيتو(octx->ctx->work_قطار, gated_delta_net_f32_tg_thread, &gctx, n_ڌاڳا);
    } وإلا {
        work_قطار_لاڳيتو(octx->ctx->work_قطار, gated_delta_net_f32_pp_thread, &gctx, n_ڌاڳا);
    }

    أرجع HTP_حالت_OK;
}
