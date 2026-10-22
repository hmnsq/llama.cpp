#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hvx-copy.h"
#అన్తర్భావ "hvx-reduce.h"
#అన్తర్భావ "hvx-exp.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp-vtcm.h"
#అన్తర్భావ "hmx-utils.h"
#అన్తర్భావ "hmx-fa-kernels.h"
#అన్తర్భావ "hmx-queue.h"
#అన్తర్భావ "gated-delta-net-ops.h"

#నిర్వచితం_కాకపోతే MIN
#నిర్వచించు MIN(a, b) ((a) < (b) ? (a) : (b))
#అయితే_ముగింపు

నిర్మాణం htp_gdn_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams;
    నిర్మాణం htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_పునాది;
    uint32_t అడ్డువరుస_ప్రారంభం;
    uint32_t nrows;
};

స్థిర అంతఃవరుస HVX_సదిశ gdn_mul_dot_f32(దశాంశం * పరిమిత dst, స్థిరరాశి HVX_సదిశ * పరిమిత mul, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc = Q6_V_vzero();
    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vd   = hvx_vmemu(dst + i * epv);
        HVX_సదిశ vm   = mul[i];
        HVX_సదిశ vdot = dot[i];
        HVX_సదిశ బహిర్గతం  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = బహిర్గతం;
        acc = hvx_vec_కూడు_f32_f32(acc, hvx_vec_mul_f32_f32(బహిర్గతం, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vm = mul[nvec];
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ బహిర్గతం = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * పరిమాణం(దశాంశం), బహిర్గతం);
        acc = hvx_vec_కూడు_f32_f32(acc, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(బహిర్గతం, vdot), zero));
    }

    తిరిగి hvx_vec_reduce_కూడిక_f32(acc);
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_mul_scalar_dot_f32(దశాంశం * పరిమిత dst, HVX_సదిశ vmul, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc = Q6_V_vzero();
    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vd   = hvx_vmemu(dst + i * epv);
        HVX_సదిశ vdot = dot[i];
        HVX_సదిశ బహిర్గతం  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = బహిర్గతం;
        acc = hvx_vec_కూడు_f32_f32(acc, hvx_vec_mul_f32_f32(బహిర్గతం, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ బహిర్గతం = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * పరిమాణం(దశాంశం), బహిర్గతం);
        acc = hvx_vec_కూడు_f32_f32(acc, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(బహిర్గతం, vdot), zero));
    }

    తిరిగి hvx_vec_reduce_కూడిక_f32(acc);
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_కూడు_scaled_dot_f32(దశాంశం * పరిమిత dst, స్థిరరాశి HVX_సదిశ * పరిమిత src,
        HVX_సదిశ vscale, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc = Q6_V_vzero();
    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vd   = hvx_vmemu(dst + i * epv);
        HVX_సదిశ vs   = src[i];
        HVX_సదిశ vdot = dot[i];
        HVX_సదిశ బహిర్గతం  = hvx_vec_కూడు_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = బహిర్గతం;
        acc = hvx_vec_కూడు_f32_f32(acc, hvx_vec_mul_f32_f32(బహిర్గతం, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vs   = src[nvec];
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ బహిర్గతం = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * పరిమాణం(దశాంశం), బహిర్గతం);
        acc = hvx_vec_కూడు_f32_f32(acc, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(బహిర్గతం, vdot), zero));
    }

    తిరిగి hvx_vec_reduce_కూడిక_f32(acc);
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_mul_dot4_f32(దశాంశం * పరిమిత dst0, దశాంశం * పరిమిత dst1,
        దశాంశం * పరిమిత dst2, దశాంశం * పరిమిత dst3,
        స్థిరరాశి HVX_సదిశ * పరిమిత mul, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc0 = Q6_V_vzero();
    HVX_సదిశ acc1 = Q6_V_vzero();
    HVX_సదిశ acc2 = Q6_V_vzero();
    HVX_సదిశ acc3 = Q6_V_vzero();

    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vm   = mul[i];
        HVX_సదిశ vdot = dot[i];

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_కూడు_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vm = mul[nvec];
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * పరిమాణం(దశాంశం), out0);
        hvx_vec_store_u(dst1 + off, nloe * పరిమాణం(దశాంశం), out1);
        hvx_vec_store_u(dst2 + off, nloe * పరిమాణం(దశాంశం), out2);
        hvx_vec_store_u(dst3 + off, nloe * పరిమాణం(దశాంశం), out3);

        acc0 = hvx_vec_కూడు_f32_f32(acc0, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_సదిశ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    తిరిగి hvx_vec_reduce_కూడిక_f32x4(acc);
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_mul_scalar_dot4_f32(దశాంశం * పరిమిత dst0, దశాంశం * పరిమిత dst1,
        దశాంశం * పరిమిత dst2, దశాంశం * పరిమిత dst3,
        HVX_సదిశ vmul, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc0 = Q6_V_vzero();
    HVX_సదిశ acc1 = Q6_V_vzero();
    HVX_సదిశ acc2 = Q6_V_vzero();
    HVX_సదిశ acc3 = Q6_V_vzero();

    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vdot = dot[i];

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_కూడు_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * పరిమాణం(దశాంశం), out0);
        hvx_vec_store_u(dst1 + off, nloe * పరిమాణం(దశాంశం), out1);
        hvx_vec_store_u(dst2 + off, nloe * పరిమాణం(దశాంశం), out2);
        hvx_vec_store_u(dst3 + off, nloe * పరిమాణం(దశాంశం), out3);

        acc0 = hvx_vec_కూడు_f32_f32(acc0, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_సదిశ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    తిరిగి hvx_vec_reduce_కూడిక_f32x4(acc);
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_కూడు_scaled_dot4_f32(దశాంశం * పరిమిత dst0, దశాంశం * పరిమిత dst1,
        దశాంశం * పరిమిత dst2, దశాంశం * పరిమిత dst3,
        స్థిరరాశి HVX_సదిశ * పరిమిత src, స్థిరరాశి దశాంశం * పరిమిత scale,
        స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc0 = Q6_V_vzero();
    HVX_సదిశ acc1 = Q6_V_vzero();
    HVX_సదిశ acc2 = Q6_V_vzero();
    HVX_సదిశ acc3 = Q6_V_vzero();
    స్థిరరాశి HVX_సదిశ scale0 = hvx_vec_splat_f32(scale[0]);
    స్థిరరాశి HVX_సదిశ scale1 = hvx_vec_splat_f32(scale[1]);
    స్థిరరాశి HVX_సదిశ scale2 = hvx_vec_splat_f32(scale[2]);
    స్థిరరాశి HVX_సదిశ scale3 = hvx_vec_splat_f32(scale[3]);

    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vs   = src[i];
        HVX_సదిశ vdot = dot[i];

        HVX_సదిశ out0 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_సదిశ out1 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_సదిశ out2 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_సదిశ out3 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_కూడు_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vs   = src[nvec];
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ out0 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_సదిశ out1 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_సదిశ out2 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_సదిశ out3 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * పరిమాణం(దశాంశం), out0);
        hvx_vec_store_u(dst1 + off, nloe * పరిమాణం(దశాంశం), out1);
        hvx_vec_store_u(dst2 + off, nloe * పరిమాణం(దశాంశం), out2);
        hvx_vec_store_u(dst3 + off, nloe * పరిమాణం(దశాంశం), out3);

        acc0 = hvx_vec_కూడు_f32_f32(acc0, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_సదిశ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    తిరిగి hvx_vec_reduce_కూడిక_f32x4(acc);
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_mul_dot8_f32(దశాంశం * పరిమిత dst0, దశాంశం * పరిమిత dst1,
        దశాంశం * పరిమిత dst2, దశాంశం * పరిమిత dst3, దశాంశం * పరిమిత dst4,
        దశాంశం * పరిమిత dst5, దశాంశం * పరిమిత dst6, దశాంశం * పరిమిత dst7,
        స్థిరరాశి HVX_సదిశ * పరిమిత mul, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc0 = Q6_V_vzero();
    HVX_సదిశ acc1 = Q6_V_vzero();
    HVX_సదిశ acc2 = Q6_V_vzero();
    HVX_సదిశ acc3 = Q6_V_vzero();
    HVX_సదిశ acc4 = Q6_V_vzero();
    HVX_సదిశ acc5 = Q6_V_vzero();
    HVX_సదిశ acc6 = Q6_V_vzero();
    HVX_సదిశ acc7 = Q6_V_vzero();

    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vm   = mul[i];
        HVX_సదిశ vdot = dot[i];

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_సదిశ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_సదిశ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_సదిశ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_సదిశ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_కూడు_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_కూడు_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_కూడు_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_కూడు_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_కూడు_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vm = mul[nvec];
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_సదిశ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_సదిశ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_సదిశ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_సదిశ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * పరిమాణం(దశాంశం), out0);
        hvx_vec_store_u(dst1 + off, nloe * పరిమాణం(దశాంశం), out1);
        hvx_vec_store_u(dst2 + off, nloe * పరిమాణం(దశాంశం), out2);
        hvx_vec_store_u(dst3 + off, nloe * పరిమాణం(దశాంశం), out3);
        hvx_vec_store_u(dst4 + off, nloe * పరిమాణం(దశాంశం), out4);
        hvx_vec_store_u(dst5 + off, nloe * పరిమాణం(దశాంశం), out5);
        hvx_vec_store_u(dst6 + off, nloe * పరిమాణం(దశాంశం), out6);
        hvx_vec_store_u(dst7 + off, nloe * పరిమాణం(దశాంశం), out7);

        acc0 = hvx_vec_కూడు_f32_f32(acc0, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_కూడు_f32_f32(acc4, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_కూడు_f32_f32(acc5, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_కూడు_f32_f32(acc6, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_కూడు_f32_f32(acc7, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_సదిశ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_సదిశ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_సదిశ rA = hvx_vec_reduce_కూడిక_f32x4(accA);
    HVX_సదిశ rB = hvx_vec_reduce_కూడిక_f32x4(accB);
    HVX_సదిశ_Pred q16 = Q6_Q_vsetq2_R(16);
    తిరిగి Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_mul_scalar_dot8_f32(దశాంశం * పరిమిత dst0, దశాంశం * పరిమిత dst1,
        దశాంశం * పరిమిత dst2, దశాంశం * పరిమిత dst3, దశాంశం * పరిమిత dst4,
        దశాంశం * పరిమిత dst5, దశాంశం * పరిమిత dst6, దశాంశం * పరిమిత dst7,
        HVX_సదిశ vmul, స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc0 = Q6_V_vzero();
    HVX_సదిశ acc1 = Q6_V_vzero();
    HVX_సదిశ acc2 = Q6_V_vzero();
    HVX_సదిశ acc3 = Q6_V_vzero();
    HVX_సదిశ acc4 = Q6_V_vzero();
    HVX_సదిశ acc5 = Q6_V_vzero();
    HVX_సదిశ acc6 = Q6_V_vzero();
    HVX_సదిశ acc7 = Q6_V_vzero();

    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vdot = dot[i];

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_సదిశ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_సదిశ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_సదిశ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_సదిశ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_కూడు_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_కూడు_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_కూడు_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_కూడు_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_కూడు_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_సదిశ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_సదిశ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_సదిశ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_సదిశ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_సదిశ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_సదిశ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_సదిశ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * పరిమాణం(దశాంశం), out0);
        hvx_vec_store_u(dst1 + off, nloe * పరిమాణం(దశాంశం), out1);
        hvx_vec_store_u(dst2 + off, nloe * పరిమాణం(దశాంశం), out2);
        hvx_vec_store_u(dst3 + off, nloe * పరిమాణం(దశాంశం), out3);
        hvx_vec_store_u(dst4 + off, nloe * పరిమాణం(దశాంశం), out4);
        hvx_vec_store_u(dst5 + off, nloe * పరిమాణం(దశాంశం), out5);
        hvx_vec_store_u(dst6 + off, nloe * పరిమాణం(దశాంశం), out6);
        hvx_vec_store_u(dst7 + off, nloe * పరిమాణం(దశాంశం), out7);

        acc0 = hvx_vec_కూడు_f32_f32(acc0, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_కూడు_f32_f32(acc4, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_కూడు_f32_f32(acc5, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_కూడు_f32_f32(acc6, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_కూడు_f32_f32(acc7, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_సదిశ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_సదిశ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_సదిశ rA = hvx_vec_reduce_కూడిక_f32x4(accA);
    HVX_సదిశ rB = hvx_vec_reduce_కూడిక_f32x4(accB);
    HVX_సదిశ_Pred q16 = Q6_Q_vsetq2_R(16);
    తిరిగి Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

స్థిర అంతఃవరుస HVX_సదిశ gdn_కూడు_scaled_dot8_f32(దశాంశం * పరిమిత dst0, దశాంశం * పరిమిత dst1,
        దశాంశం * పరిమిత dst2, దశాంశం * పరిమిత dst3, దశాంశం * పరిమిత dst4,
        దశాంశం * పరిమిత dst5, దశాంశం * పరిమిత dst6, దశాంశం * పరిమిత dst7,
        స్థిరరాశి HVX_సదిశ * పరిమిత src, స్థిరరాశి దశాంశం * పరిమిత scale,
        స్థిరరాశి HVX_సదిశ * పరిమిత dot, uint32_t n) {
    HVX_సదిశ acc0 = Q6_V_vzero();
    HVX_సదిశ acc1 = Q6_V_vzero();
    HVX_సదిశ acc2 = Q6_V_vzero();
    HVX_సదిశ acc3 = Q6_V_vzero();
    HVX_సదిశ acc4 = Q6_V_vzero();
    HVX_సదిశ acc5 = Q6_V_vzero();
    HVX_సదిశ acc6 = Q6_V_vzero();
    HVX_సదిశ acc7 = Q6_V_vzero();
    స్థిరరాశి HVX_సదిశ scale0 = hvx_vec_splat_f32(scale[0]);
    స్థిరరాశి HVX_సదిశ scale1 = hvx_vec_splat_f32(scale[1]);
    స్థిరరాశి HVX_సదిశ scale2 = hvx_vec_splat_f32(scale[2]);
    స్థిరరాశి HVX_సదిశ scale3 = hvx_vec_splat_f32(scale[3]);
    స్థిరరాశి HVX_సదిశ scale4 = hvx_vec_splat_f32(scale[4]);
    స్థిరరాశి HVX_సదిశ scale5 = hvx_vec_splat_f32(scale[5]);
    స్థిరరాశి HVX_సదిశ scale6 = hvx_vec_splat_f32(scale[6]);
    స్థిరరాశి HVX_సదిశ scale7 = hvx_vec_splat_f32(scale[7]);

    స్థిరరాశి uint32_t epv = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;
    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        HVX_సదిశ vs   = src[i];
        HVX_సదిశ vdot = dot[i];

        HVX_సదిశ out0 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_సదిశ out1 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_సదిశ out2 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_సదిశ out3 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_సదిశ out4 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_సదిశ out5 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_సదిశ out6 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_సదిశ out7 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_కూడు_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_కూడు_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_కూడు_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_కూడు_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_కూడు_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ఒకవేళ (nloe) {
        స్థిరరాశి uint32_t off = nvec * epv;
        HVX_సదిశ vs   = src[nvec];
        HVX_సదిశ vdot = dot[nvec];
        HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ zero = Q6_V_vzero();

        HVX_సదిశ out0 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_సదిశ out1 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_సదిశ out2 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_సదిశ out3 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_సదిశ out4 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_సదిశ out5 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_సదిశ out6 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_సదిశ out7 = hvx_vec_కూడు_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * పరిమాణం(దశాంశం), out0);
        hvx_vec_store_u(dst1 + off, nloe * పరిమాణం(దశాంశం), out1);
        hvx_vec_store_u(dst2 + off, nloe * పరిమాణం(దశాంశం), out2);
        hvx_vec_store_u(dst3 + off, nloe * పరిమాణం(దశాంశం), out3);
        hvx_vec_store_u(dst4 + off, nloe * పరిమాణం(దశాంశం), out4);
        hvx_vec_store_u(dst5 + off, nloe * పరిమాణం(దశాంశం), out5);
        hvx_vec_store_u(dst6 + off, nloe * పరిమాణం(దశాంశం), out6);
        hvx_vec_store_u(dst7 + off, nloe * పరిమాణం(దశాంశం), out7);

        acc0 = hvx_vec_కూడు_f32_f32(acc0, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_కూడు_f32_f32(acc1, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_కూడు_f32_f32(acc2, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_కూడు_f32_f32(acc3, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_కూడు_f32_f32(acc4, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_కూడు_f32_f32(acc5, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_కూడు_f32_f32(acc6, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_కూడు_f32_f32(acc7, Q6_V_vmux_QVV(ముసుగు, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_సదిశ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_సదిశ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_సదిశ rA = hvx_vec_reduce_కూడిక_f32x4(accA);
    HVX_సదిశ rB = hvx_vec_reduce_కూడిక_f32x4(accB);
    HVX_సదిశ_Pred q16 = Q6_Q_vsetq2_R(16);
    తిరిగి Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

స్థిర అంతఃవరుస శూన్య gdn_అడుగు_kda_f32(
    దశాంశం * పరిమిత s_work,
    దశాంశం * పరిమిత attn_బహిర్గతం,
    స్థిరరాశి దశాంశం * పరిమిత q_t,
    స్థిరరాశి దశాంశం * పరిమిత k_t,
    స్థిరరాశి దశాంశం * పరిమిత v_t,
    స్థిరరాశి దశాంశం * పరిమిత g_t,
    దశాంశం beta_val,
    దశాంశం scale,
    uint32_t S_v
) {
    స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = S_v / epv;
    స్థిరరాశి uint32_t nloe = S_v % epv;

    HVX_సదిశ vq[4];
    HVX_సదిశ vk[4];
    HVX_సదిశ vg[4];

    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    ఒకవేళ (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
    }

    స్థిరరాశి HVX_సదిశ vbeta  = hvx_vec_splat_f32(beta_val);
    స్థిరరాశి HVX_సదిశ vscale = hvx_vec_splat_f32(scale);

    దశాంశం delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    కొరకు (; j + 8 <= S_v; j += 8) {
        దశాంశం * row0 = s_work + (uint64_t) (j + 0) * S_v;
        దశాంశం * row1 = s_work + (uint64_t) (j + 1) * S_v;
        దశాంశం * row2 = s_work + (uint64_t) (j + 2) * S_v;
        దశాంశం * row3 = s_work + (uint64_t) (j + 3) * S_v;
        దశాంశం * row4 = s_work + (uint64_t) (j + 4) * S_v;
        దశాంశం * row5 = s_work + (uint64_t) (j + 5) * S_v;
        దశాంశం * row6 = s_work + (uint64_t) (j + 6) * S_v;
        దశాంశం * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_సదిశ vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_సదిశ vv_t   = hvx_vmemu(v_t + j);
        HVX_సదిశ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_సదిశ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * పరిమాణం(దశాంశం), vdelta);

        HVX_సదిశ vattn = gdn_కూడు_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_సదిశ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_బహిర్గతం + j, 8 * పరిమాణం(దశాంశం), res_attn);
    }
    కొరకు (; j + 4 <= S_v; j += 4) {
        దశాంశం * row0 = s_work + (uint64_t) (j + 0) * S_v;
        దశాంశం * row1 = s_work + (uint64_t) (j + 1) * S_v;
        దశాంశం * row2 = s_work + (uint64_t) (j + 2) * S_v;
        దశాంశం * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_సదిశ vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_సదిశ vv_t   = hvx_vmemu(v_t + j);
        HVX_సదిశ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_సదిశ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * పరిమాణం(దశాంశం), vdelta);

        HVX_సదిశ vattn = gdn_కూడు_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_సదిశ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_బహిర్గతం + j, 4 * పరిమాణం(దశాంశం), res_attn);
    }
    కొరకు (; j < S_v; ++j) {
        దశాంశం * అడ్డువరుస = s_work + (uint64_t) j * S_v;
        HVX_సదిశ vsum = gdn_mul_dot_f32(అడ్డువరుస, vg, vk, S_v);
        HVX_సదిశ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_సదిశ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_సదిశ vres = gdn_కూడు_scaled_dot_f32(అడ్డువరుస, vk, vdj, vq, S_v);
        attn_బహిర్గతం[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

స్థిర అంతఃవరుస శూన్య gdn_అడుగు_scalar_f32(
    దశాంశం * పరిమిత s_work,
    దశాంశం * పరిమిత attn_బహిర్గతం,
    స్థిరరాశి దశాంశం * పరిమిత q_t,
    స్థిరరాశి దశాంశం * పరిమిత k_t,
    స్థిరరాశి దశాంశం * పరిమిత v_t,
    స్థిరరాశి దశాంశం * పరిమిత g_t,
    దశాంశం beta_val,
    దశాంశం scale,
    uint32_t S_v
) {
    స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = S_v / epv;
    స్థిరరాశి uint32_t nloe = S_v % epv;

    HVX_సదిశ vq[4];
    HVX_సదిశ vk[4];

    కొరకు (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    ఒకవేళ (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    స్థిరరాశి HVX_సదిశ vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    స్థిరరాశి HVX_సదిశ vbeta  = hvx_vec_splat_f32(beta_val);
    స్థిరరాశి HVX_సదిశ vscale = hvx_vec_splat_f32(scale);

    దశాంశం delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    కొరకు (; j + 8 <= S_v; j += 8) {
        దశాంశం * row0 = s_work + (uint64_t) (j + 0) * S_v;
        దశాంశం * row1 = s_work + (uint64_t) (j + 1) * S_v;
        దశాంశం * row2 = s_work + (uint64_t) (j + 2) * S_v;
        దశాంశం * row3 = s_work + (uint64_t) (j + 3) * S_v;
        దశాంశం * row4 = s_work + (uint64_t) (j + 4) * S_v;
        దశాంశం * row5 = s_work + (uint64_t) (j + 5) * S_v;
        దశాంశం * row6 = s_work + (uint64_t) (j + 6) * S_v;
        దశాంశం * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_సదిశ vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_సదిశ vv_t   = hvx_vmemu(v_t + j);
        HVX_సదిశ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_సదిశ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * పరిమాణం(దశాంశం), vdelta);

        HVX_సదిశ vattn = gdn_కూడు_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_సదిశ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_బహిర్గతం + j, 8 * పరిమాణం(దశాంశం), res_attn);
    }
    కొరకు (; j + 4 <= S_v; j += 4) {
        దశాంశం * row0 = s_work + (uint64_t) (j + 0) * S_v;
        దశాంశం * row1 = s_work + (uint64_t) (j + 1) * S_v;
        దశాంశం * row2 = s_work + (uint64_t) (j + 2) * S_v;
        దశాంశం * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_సదిశ vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_సదిశ vv_t   = hvx_vmemu(v_t + j);
        HVX_సదిశ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_సదిశ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * పరిమాణం(దశాంశం), vdelta);

        HVX_సదిశ vattn = gdn_కూడు_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_సదిశ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_బహిర్గతం + j, 4 * పరిమాణం(దశాంశం), res_attn);
    }
    కొరకు (; j < S_v; ++j) {
        దశాంశం * అడ్డువరుస = s_work + (uint64_t) j * S_v;
        HVX_సదిశ vsum = gdn_mul_scalar_dot_f32(అడ్డువరుస, vgate, vk, S_v);
        HVX_సదిశ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_సదిశ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_సదిశ vres = gdn_కూడు_scaled_dot_f32(అడ్డువరుస, vk, vdj, vq, S_v);
        attn_బహిర్గతం[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

స్థిర శూన్య gated_delta_net_f32_pp_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_gdn_context * gctx = (నిర్మాణం htp_gdn_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = gctx->octx;
    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams = gctx->kparams;

    స్థిరరాశి నిర్మాణం htp_tensor * q     = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k     = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v     = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * g     = octx->src[3];
    స్థిరరాశి నిర్మాణం htp_tensor * beta  = octx->src[4];
    స్థిరరాశి నిర్మాణం htp_tensor * state = octx->src[5];
    స్థిరరాశి నిర్మాణం htp_tensor * dst   = octx->dst;

    స్థిరరాశి uint32_t S_v      = kparams->S_v;
    స్థిరరాశి uint32_t H        = kparams->H;
    స్థిరరాశి uint32_t n_టోకెన్లు = kparams->n_టోకెన్లు;
    స్థిరరాశి uint32_t n_seqs   = kparams->n_seqs;
    స్థిరరాశి uint32_t K        = kparams->K;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర  = gctx->అడ్డువరుస_ప్రారంభం + gctx->nrows;

    ఒకవేళ (ith >= gctx->nrows) {
        తిరిగి;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst_cache = octx->dsts[1];
    స్థిరరాశి దశాంశం scale = kparams->scale;
    దశాంశం * dst_పునాది       = (దశాంశం *) (uintptr_t) dst->దత్తాంశం;
    దశాంశం * state_బహిర్గతం_పునాది = dst_cache ? (దశాంశం *) (uintptr_t) dst_cache->దత్తాంశం : (dst_పునాది + S_v * H * n_టోకెన్లు * n_seqs);

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    స్థిరరాశి నిర్మాణం htp_gdn_vtcm_layout * layout = &gctx->layout;
    దశాంశం * s_work[2];
    s_work[0] = (దశాంశం *) (gctx->vtcm_పునాది + layout->బైట్లు_చొప్పున_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / పరిమాణం(దశాంశం);

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_H   = &kparams->div_H;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_q1  = &kparams->div_q1;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_k1  = &kparams->div_k1;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_rq3 = &kparams->div_rq3;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_rk3 = &kparams->div_rk3;

    స్థిరరాశి uint32_t state_seq_stride = kparams->state_seq_stride;
    స్థిరరాశి uint64_t state_కొలత_చొప్పున_snap = (uint64_t) kparams->state_కొలత_చొప్పున_snap;
    స్థిరరాశి dma_addr_t state_బహిర్గతం_dma_పునాది = dst_cache ? dst_cache->దత్తాంశం : (dst->దత్తాంశం + S_v * H * n_టోకెన్లు * n_seqs * పరిమాణం(దశాంశం));

    uint32_t ir_prefetch = gctx->అడ్డువరుస_ప్రారంభం + ith;
    పూర్ణాంకం spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    కొరకు (పూర్ణాంకం అడుగు = 0; అడుగు < 2 && ir_prefetch < అడ్డువరుస_చివర; అడుగు++) {
        స్థిరరాశి uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        స్థిరరాశి uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_లోపల  = state->దత్తాంశం + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * పరిమాణం(దశాంశం);
        dma_addr_t ps_బహిర్గతం = state_బహిర్గతం_dma_పునాది + ((uint64_t) piv3 * H + piv1) * S_v * S_v * పరిమాణం(దశాంశం);

        // Push dummy write-back
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ps_బహిర్గతం, s_work[spad_idx]),
                       S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                       S_v * పరిమాణం(దశాంశం), 0);

        // Push fetch
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_work[spad_idx], ps_లోపల),
                       S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                       S_v * పరిమాణం(దశాంశం), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    పూర్ణాంకం curr_spad_idx = 0;
    కొరకు (uint32_t ir = gctx->అడ్డువరుస_ప్రారంభం + ith; ir < అడ్డువరుస_చివర; ir += nth) {
        dma_వరుస_తీయి(dma_q);
        dma_వరుస_తీయి(dma_q);

        దశాంశం * s_work_curr = s_work[curr_spad_idx];

        స్థిరరాశి uint32_t iv1 = fastmodulo(ir, H, fd_H);
        స్థిరరాశి uint32_t iv3 = fastdiv(ir, fd_H);

        స్థిరరాశి uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        స్థిరరాశి uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        స్థిరరాశి uint32_t iq3 = fastdiv(iv3, fd_rq3);
        స్థిరరాశి uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_బహిర్గతం  = state_బహిర్గతం_dma_పునాది + ((uint64_t) iv3 * H + iv1) * S_v * S_v * పరిమాణం(దశాంశం);
        దశాంశం * attn_దత్తాంశం = dst_పునాది + ((uint64_t) iv3 * n_టోకెన్లు * H + iv1) * S_v;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        కొరకు (uint32_t t = 0; t < n_టోకెన్లు; ++t) {
            స్థిరరాశి దశాంశం * q_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) q->దత్తాంశం +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            స్థిరరాశి దశాంశం * k_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) k->దత్తాంశం +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            స్థిరరాశి దశాంశం * v_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) v->దత్తాంశం +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            స్థిరరాశి దశాంశం * g_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) g->దత్తాంశం +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            స్థిరరాశి దశాంశం beta_val = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) beta->దత్తాంశం +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            ఒకవేళ (kparams->kda) {
                gdn_అడుగు_kda_f32(s_work_curr, attn_దత్తాంశం, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } లేకపోతే {
                gdn_అడుగు_scalar_f32(s_work_curr, attn_దత్తాంశం, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            ఒకవేళ (K > 1) {
                స్థిరరాశి int64_t లక్ష్యం_స్థానం = (int64_t) n_టోకెన్లు - 1 - (int64_t) t;
                ఒకవేళ (లక్ష్యం_స్థానం > 0 && లక్ష్యం_స్థానం < (int64_t) K) {
                    దశాంశం * curr_state_o = state_బహిర్గతం_పునాది + (uint64_t) లక్ష్యం_స్థానం * state_కొలత_చొప్పున_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (స్థిరరాశి uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_దత్తాంశం += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_బహిర్గతం, s_work_curr),
                       S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                       S_v * పరిమాణం(దశాంశం), S_v);

        // Prefetch next block (if any)
        ఒకవేళ (ir_prefetch < అడ్డువరుస_చివర) {
            స్థిరరాశి uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            స్థిరరాశి uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_లోపల = state->దత్తాంశం + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * పరిమాణం(దశాంశం);

            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_work[spad_idx], ps_లోపల),
                           S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                           S_v * పరిమాణం(దశాంశం), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_వరుస_flush(dma_q);
}

స్థిర శూన్య gated_delta_net_f32_tg_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_gdn_context * gctx = (నిర్మాణం htp_gdn_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = gctx->octx;
    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams = gctx->kparams;

    స్థిరరాశి నిర్మాణం htp_tensor * q     = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k     = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v     = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * g     = octx->src[3];
    స్థిరరాశి నిర్మాణం htp_tensor * beta  = octx->src[4];
    స్థిరరాశి నిర్మాణం htp_tensor * state = octx->src[5];
    స్థిరరాశి నిర్మాణం htp_tensor * dst   = octx->dst;

    స్థిరరాశి uint32_t S_v      = kparams->S_v;
    స్థిరరాశి uint32_t H        = kparams->H;
    స్థిరరాశి uint32_t n_seqs   = kparams->n_seqs;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర  = gctx->అడ్డువరుస_ప్రారంభం + gctx->nrows;

    ఒకవేళ (ith >= gctx->nrows) {
        తిరిగి;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst_cache = octx->dsts[1];
    స్థిరరాశి దశాంశం scale = kparams->scale;
    దశాంశం * dst_పునాది  = (దశాంశం *) (uintptr_t) dst->దత్తాంశం;

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    స్థిరరాశి నిర్మాణం htp_gdn_vtcm_layout * layout = &gctx->layout;
    దశాంశం * s_work[2];
    s_work[0] = (దశాంశం *) (gctx->vtcm_పునాది + layout->బైట్లు_చొప్పున_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / పరిమాణం(దశాంశం);

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_H   = &kparams->div_H;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_q1  = &kparams->div_q1;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_k1  = &kparams->div_k1;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_rq3 = &kparams->div_rq3;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * fd_rk3 = &kparams->div_rk3;

    స్థిరరాశి uint32_t state_seq_stride = kparams->state_seq_stride;
    స్థిరరాశి dma_addr_t state_బహిర్గతం_dma_పునాది = dst_cache ? dst_cache->దత్తాంశం : (dst->దత్తాంశం + S_v * H * n_seqs * పరిమాణం(దశాంశం));

    uint32_t ir_prefetch = gctx->అడ్డువరుస_ప్రారంభం + ith;
    పూర్ణాంకం spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    కొరకు (పూర్ణాంకం అడుగు = 0; అడుగు < 2 && ir_prefetch < అడ్డువరుస_చివర; అడుగు++) {
        స్థిరరాశి uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        స్థిరరాశి uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_లోపల  = state->దత్తాంశం + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * పరిమాణం(దశాంశం);
        dma_addr_t ps_బహిర్గతం = state_బహిర్గతం_dma_పునాది + ((uint64_t) piv3 * H + piv1) * S_v * S_v * పరిమాణం(దశాంశం);

        // Push dummy write-back
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ps_బహిర్గతం, s_work[spad_idx]),
                       S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                       S_v * పరిమాణం(దశాంశం), 0);

        // Push fetch
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_work[spad_idx], ps_లోపల),
                       S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                       S_v * పరిమాణం(దశాంశం), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    పూర్ణాంకం curr_spad_idx = 0;
    కొరకు (uint32_t ir = gctx->అడ్డువరుస_ప్రారంభం + ith; ir < అడ్డువరుస_చివర; ir += nth) {
        dma_వరుస_తీయి(dma_q);
        dma_వరుస_తీయి(dma_q);

        దశాంశం * s_work_curr = s_work[curr_spad_idx];

        స్థిరరాశి uint32_t iv1 = fastmodulo(ir, H, fd_H);
        స్థిరరాశి uint32_t iv3 = fastdiv(ir, fd_H);

        స్థిరరాశి uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        స్థిరరాశి uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        స్థిరరాశి uint32_t iq3 = fastdiv(iv3, fd_rq3);
        స్థిరరాశి uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_బహిర్గతం  = state_బహిర్గతం_dma_పునాది + ((uint64_t) iv3 * H + iv1) * S_v * S_v * పరిమాణం(దశాంశం);
        దశాంశం * attn_దత్తాంశం = dst_పునాది + ((uint64_t) iv3 * H + iv1) * S_v;

        స్థిరరాశి దశాంశం * q_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) q->దత్తాంశం +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        స్థిరరాశి దశాంశం * k_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) k->దత్తాంశం +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        స్థిరరాశి దశాంశం * v_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) v->దత్తాంశం +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        స్థిరరాశి దశాంశం * g_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) g->దత్తాంశం +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        స్థిరరాశి దశాంశం beta_val = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) beta->దత్తాంశం +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ఒకవేళ (kparams->kda) {
            gdn_అడుగు_kda_f32(s_work_curr, attn_దత్తాంశం, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } లేకపోతే {
            gdn_అడుగు_scalar_f32(s_work_curr, attn_దత్తాంశం, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_బహిర్గతం, s_work_curr),
                       S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                       S_v * పరిమాణం(దశాంశం), S_v);

        // Prefetch next block (if any)
        ఒకవేళ (ir_prefetch < అడ్డువరుస_చివర) {
            స్థిరరాశి uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            స్థిరరాశి uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_లోపల = state->దత్తాంశం + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * పరిమాణం(దశాంశం);

            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_work[spad_idx], ps_లోపల),
                           S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం),
                           S_v * పరిమాణం(దశాంశం), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_వరుస_flush(dma_q);
}

నిర్మాణం htp_gdn_hmx_gemm_task {
    స్థిరరాశి __fp16 * అడ్డువరుస_tiles;
    స్థిరరాశి __fp16 * నిలువు_tiles;
    __fp16 *       బహిర్గతం_tiles;
    uint32_t       n_అడ్డువరుస_tiles;
    uint32_t       n_నిలువు_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

స్థిర శూన్య htp_gdn_hmx_gemm_కార్మికుడు(శూన్య * దత్తాంశం) {
    నిర్మాణం htp_gdn_hmx_gemm_task * task = (నిర్మాణం htp_gdn_hmx_gemm_task *) దత్తాంశం;
    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)task->hmx_scales));

    స్థిరరాశి size_t dot_stride = task->dot_stride;
    కొరకు (uint32_t r = 0; r < task->n_అడ్డువరుస_tiles; ++r) {
        స్థిరరాశి __fp16 * r_tiles = task->అడ్డువరుస_tiles + r * dot_stride;
        స్థిరరాశి __fp16 * c_tiles = task->నిలువు_tiles;
        __fp16 *       o_tile  = task->బహిర్గతం_tiles + r * task->n_నిలువు_tiles * HMX_FP16_TILE_N_ELMS;

        కొరకు (uint32_t c = 0; c < task->n_నిలువు_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

స్థిర అంతఃవరుస శూన్య htp_gdn_నెట్టు_hmx_gemm_task(
    hmx_వరుస_t q,
    నిర్మాణం htp_gdn_hmx_gemm_task * task,
    స్థిరరాశి __fp16 * అడ్డువరుస_tiles,
    స్థిరరాశి __fp16 * నిలువు_tiles,
    __fp16 * బహిర్గతం_tiles,
    uint32_t n_అడ్డువరుస_tiles,
    uint32_t n_నిలువు_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    task->అడ్డువరుస_tiles   = అడ్డువరుస_tiles;
    task->నిలువు_tiles   = నిలువు_tiles;
    task->బహిర్గతం_tiles   = బహిర్గతం_tiles;
    task->n_అడ్డువరుస_tiles = n_అడ్డువరుస_tiles;
    task->n_నిలువు_tiles = n_నిలువు_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    task->hmx_scales  = scales;

    hmx_వరుస_నెట్టు(q, hmx_వరుస_make_desc(htp_gdn_hmx_gemm_కార్మికుడు, task));
}

స్థిర అంతఃవరుస శూన్య gdn_unpack_64x64_tiles_గమ్యం_vectors(
    HVX_సదిశ * పరిమిత అడ్డువరుసలు,
    స్థిరరాశి __fp16 * పరిమిత tiles
) {
    స్థిరరాశి HVX_సదిశ * t00 = (స్థిరరాశి HVX_సదిశ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    స్థిరరాశి HVX_సదిశ * t01 = (స్థిరరాశి HVX_సదిశ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    స్థిరరాశి HVX_సదిశ * t10 = (స్థిరరాశి HVX_సదిశ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    స్థిరరాశి HVX_సదిశ * t11 = (స్థిరరాశి HVX_సదిశ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        HVX_సదిశ_Pair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        అడ్డువరుసలు[2 * r + 0] = Q6_V_lo_W(vp0);
        అడ్డువరుసలు[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_సదిశ_Pair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        అడ్డువరుసలు[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        అడ్డువరుసలు[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

స్థిర అంతఃవరుస శూన్య gdn_pack_64x64_vectors_గమ్యం_tiles(
    __fp16 * పరిమిత tiles,
    స్థిరరాశి HVX_సదిశ * పరిమిత అడ్డువరుసలు
) {
    HVX_సదిశ * t00 = (HVX_సదిశ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_సదిశ * t01 = (HVX_సదిశ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_సదిశ * t10 = (HVX_సదిశ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_సదిశ * t11 = (HVX_సదిశ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        HVX_సదిశ_Pair vp0 = Q6_W_vshuff_VVR(అడ్డువరుసలు[2 * r + 1], అడ్డువరుసలు[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_సదిశ_Pair vp1 = Q6_W_vshuff_VVR(అడ్డువరుసలు[32 + 2 * r + 1], అడ్డువరుసలు[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

స్థిర అంతఃవరుస శూన్య gdn_unpack_64x_S_tiles_గమ్యం_f32(
    దశాంశం * పరిమిత dst_f32,
    స్థిరరాశి __fp16 * పరిమిత tiles,
    uint32_t S_v
) {
    స్థిరరాశి uint32_t n_నిలువు_tiles = S_v / 32;
    కొరకు (uint32_t r0 = 0; r0 < 2; ++r0) {
        కొరకు (uint32_t d = 0; d < S_v / 64; ++d) {
            స్థిరరాశి HVX_సదిశ * t0 = (స్థిరరాశి HVX_సదిశ *) (tiles + (r0 * n_నిలువు_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            స్థిరరాశి HVX_సదిశ * t1 = (స్థిరరాశి HVX_సదిశ *) (tiles + (r0 * n_నిలువు_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            కొరకు (uint32_t r = 0; r < 16; ++r) {
                HVX_సదిశ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_సదిశ_Pair p0 = hvx_vec_f16_గమ్యం_f32(Q6_V_lo_W(vp01));
                HVX_సదిశ_Pair p1 = hvx_vec_f16_గమ్యం_f32(Q6_V_hi_W(vp01));

                దశాంశం * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                దశాంశం * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య gdn_unpack_64x_S_tiles_గమ్యం_f16(
    __fp16 * పరిమిత dst_f16,
    స్థిరరాశి __fp16 * పరిమిత tiles,
    uint32_t S_v
) {
    స్థిరరాశి uint32_t n_నిలువు_tiles = S_v / 32;
    కొరకు (uint32_t r0 = 0; r0 < 2; ++r0) {
        కొరకు (uint32_t d = 0; d < S_v / 64; ++d) {
            స్థిరరాశి HVX_సదిశ * t0 = (స్థిరరాశి HVX_సదిశ *) (tiles + (r0 * n_నిలువు_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            స్థిరరాశి HVX_సదిశ * t1 = (స్థిరరాశి HVX_సదిశ *) (tiles + (r0 * n_నిలువు_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            కొరకు (uint32_t r = 0; r < 16; ++r) {
                HVX_సదిశ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య gdn_unpack_Sx_S_tiles_గమ్యం_f32(
    దశాంశం * పరిమిత dst_f32,
    స్థిరరాశి __fp16 * పరిమిత tiles,
    uint32_t S_v
) {
    స్థిరరాశి uint32_t n_tiles = S_v / 32;
    కొరకు (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        కొరకు (uint32_t d = 0; d < S_v / 64; ++d) {
            స్థిరరాశి HVX_సదిశ * t0 = (స్థిరరాశి HVX_సదిశ *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            స్థిరరాశి HVX_సదిశ * t1 = (స్థిరరాశి HVX_సదిశ *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            కొరకు (uint32_t r = 0; r < 16; ++r) {
                HVX_సదిశ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_సదిశ_Pair p0 = hvx_vec_f16_గమ్యం_f32(Q6_V_lo_W(vp01));
                HVX_సదిశ_Pair p1 = hvx_vec_f16_గమ్యం_f32(Q6_V_hi_W(vp01));

                దశాంశం * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                దశాంశం * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య gdn_f32_గమ్యం_hmx_అడ్డువరుస_tiles_and_f16(
    __fp16 * పరిమిత dst_tiles,
    __fp16 * పరిమిత dst_ప్రధానసంఖ్య_tiles,
    __fp16 * పరిమిత dst_f16,
    స్థిరరాశి దశాంశం * పరిమిత src,
    స్థిరరాశి __fp16 * పరిమిత scale_చొప్పున_అడ్డువరుస,
    uint32_t n_అడ్డువరుసలు,
    uint32_t n_నిలువుల_సంఖ్య
) {
    స్థిరరాశి uint32_t n_నిలువు_tiles = n_నిలువుల_సంఖ్య / 32;
    స్థిరరాశి uint32_t * scale_జతలు = (స్థిరరాశి uint32_t *) scale_చొప్పున_అడ్డువరుస;

    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        స్థిరరాశి దశాంశం * p0 = src + (r + 0) * n_నిలువుల_సంఖ్య;
        స్థిరరాశి దశాంశం * p1 = src + (r + 1) * n_నిలువుల_సంఖ్య;

        HVX_సదిశ v_scale;
        ఒకవేళ (dst_ప్రధానసంఖ్య_tiles) {
            uint32_t scale_pair = scale_జతలు ? scale_జతలు[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        కొరకు (uint32_t c = 0; c < n_నిలువు_tiles; c += 2) {
            HVX_సదిశ v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_సదిశ v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_సదిశ v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_సదిశ v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_సదిశ vh0 = hvx_vec_f32_గమ్యం_f16_shuff(v0_0, v1_0);
            HVX_సదిశ vh1 = hvx_vec_f32_గమ్యం_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_నిలువు_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_నిలువు_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_సదిశ *) tile0)[r1] = vh0;
            ((HVX_సదిశ *) tile1)[r1] = vh1;

            ఒకవేళ (dst_ప్రధానసంఖ్య_tiles) {
                HVX_సదిశ vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_సదిశ vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_ప్రధానసంఖ్య_tiles + (r0 * n_నిలువు_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_ప్రధానసంఖ్య_tiles + (r0 * n_నిలువు_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_సదిశ *) tile0_s)[r1] = vh0_s;
                ((HVX_సదిశ *) tile1_s)[r1] = vh1_s;
            }

            ఒకవేళ (dst_f16) {
                HVX_సదిశ_Pair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_నిలువుల_సంఖ్య + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_నిలువుల_సంఖ్య + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య hvx_transpose_32x32_పదాలు(HVX_సదిశ * పరిమిత m, HVX_సదిశ * పరిమిత tmp) {
    కొరకు (పూర్ణాంకం i = 0; i < 16; ++i) {
        HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    కొరకు (పూర్ణాంకం b = 0; b < 32; b += 4) {
        HVX_సదిశ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_సదిశ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    కొరకు (పూర్ణాంకం b = 0; b < 32; b += 8) {
        కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
            HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    కొరకు (పూర్ణాంకం b = 0; b < 32; b += 16) {
        కొరకు (పూర్ణాంకం i = 0; i < 8; ++i) {
            HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < 16; ++i) {
        HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

స్థిర అంతఃవరుస శూన్య gdn_pack_d_t_అడ్డువరుస_tiles(
    __fp16 * పరిమిత dst_tiles,
    స్థిరరాశి __fp16 * పరిమిత src_d,
    uint32_t S_v,
    HVX_సదిశ * పరిమిత m,
    HVX_సదిశ * పరిమిత tmp
) {
    కొరకు (uint32_t నిలువు_half = 0; నిలువు_half < S_v / 64; ++నిలువు_half) {
        uint32_t r0_పునాది = నిలువు_half * 2;
        కొరకు (uint32_t c0 = 0; c0 < 2; ++c0) {
            కొరకు (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + నిలువు_half * 64);
            }

            hvx_transpose_32x32_పదాలు(m, tmp);

            uint32_t tile0_idx = (r0_పునాది + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_పునాది + 1) * 2 + c0;
            HVX_సదిశ * t0 = (HVX_సదిశ *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_సదిశ * t1 = (HVX_సదిశ *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            కొరకు (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

స్థిర __attribute__((noinline)) శూన్య gdn_build_inv_l_ఖండాలు(
    __fp16 * పరిమిత inv_అడ్డువరుస_tiles,
    స్థిరరాశి HVX_సదిశ * పరిమిత అడ్డువరుసలు_kk,
    స్థిరరాశి __fp16 * పరిమిత decay_m,
    స్థిరరాశి దశాంశం * పరిమిత beta,
    __fp16 * పరిమిత l10_tile,
    __fp16 * పరిమిత neg_a11_tile
) {
    స్థిరరాశి HVX_సదిశ v_one_f16 = hvx_vec_splat_f16(1.0f);
    స్థిరరాశి HVX_సదిశ_Pred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_సదిశ * పరిమిత p_l00 = (HVX_సదిశ *) l00;
    HVX_సదిశ * పరిమిత p_l11 = (HVX_సదిశ *) l11;
    HVX_సదిశ * పరిమిత p_l10_tile = (HVX_సదిశ *) l10_tile;

    HVX_సదిశ * పరిమిత tile00 = (HVX_సదిశ *) (inv_అడ్డువరుస_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_సదిశ * పరిమిత tile01 = (HVX_సదిశ *) (inv_అడ్డువరుస_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_సదిశ * పరిమిత tile11 = (HVX_సదిశ *) (inv_అడ్డువరుస_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_సదిశ * పరిమిత p_neg_a11 = (HVX_సదిశ *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_గమ్యం_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_సదిశ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_సదిశ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_సదిశ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_సదిశ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_సదిశ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(అడ్డువరుసలు_kk[t0], v_d0), v_b0);
        HVX_సదిశ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(అడ్డువరుసలు_kk[t1], v_d1), v_b1);

        p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
    }

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 32 + 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_సదిశ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_సదిశ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_సదిశ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_సదిశ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_సదిశ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(అడ్డువరుసలు_kk[t0], v_d0), v_b0);
        HVX_సదిశ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(అడ్డువరుసలు_kk[t1], v_d1), v_b1);

        HVX_సదిశ_Pair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
        p_l10_tile[r] = Q6_V_lo_W(vp_l10);
        p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
    }

    HVX_సదిశ a_అడ్డువరుసలు[32];
    కొరకు (uint32_t t = 0; t < 32; ++t) {
        HVX_సదిశ v_inv = Q6_V_vzero();
        కొరకు (uint32_t k = 0; k < t; ++k) {
            HVX_సదిశ v_lk = Q6_Vh_vsplat_R(l00[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_అడ్డువరుసలు[k]));
        }
        HVX_సదిశ_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_అడ్డువరుసలు[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(a_అడ్డువరుసలు[2 * r + 1], a_అడ్డువరుసలు[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    కొరకు (uint32_t t = 0; t < 32; ++t) {
        HVX_సదిశ v_inv = Q6_V_vzero();
        కొరకు (uint32_t k = 0; k < t; ++k) {
            HVX_సదిశ v_lk = Q6_Vh_vsplat_R(l11[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_అడ్డువరుసలు[k]));
        }
        HVX_సదిశ_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_అడ్డువరుసలు[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    కొరకు (uint32_t r = 0; r < 16; ++r) {
        HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(a_అడ్డువరుసలు[2 * r + 1], a_అడ్డువరుసలు[2 * r + 0], -2);
        tile11[r] = Q6_V_lo_W(vp);

        HVX_సదిశ n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_అడ్డువరుసలు[2 * r + 0]);
        HVX_సదిశ n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_అడ్డువరుసలు[2 * r + 1]);
        HVX_సదిశ_Pair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
        p_neg_a11[r] = Q6_V_lo_W(vp_neg);
    }
}


స్థిర అంతఃవరుస శూన్య gdn_dma_నెట్టు_chunk_inputs(
    dma_వరుస * dma_q,
    దశాంశం * vtcm_q,
    దశాంశం * vtcm_k,
    దశాంశం * vtcm_v,
    స్థిరరాశి నిర్మాణం htp_tensor * q,
    స్థిరరాశి నిర్మాణం htp_tensor * k,
    స్థిరరాశి నిర్మాణం htp_tensor * v,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_కొలత,
    uint32_t S_v
) {
    స్థిరరాశి dma_addr_t q_dma = q->దత్తాంశం + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
    స్థిరరాశి dma_addr_t k_dma = k->దత్తాంశం + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
    స్థిరరాశి dma_addr_t v_dma = v->దత్తాంశం + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_q, q_dma), S_v * పరిమాణం(దశాంశం), q->nb[2], S_v * పరిమాణం(దశాంశం), chunk_కొలత);
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_k, k_dma), S_v * పరిమాణం(దశాంశం), k->nb[2], S_v * పరిమాణం(దశాంశం), chunk_కొలత);
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_v, v_dma), S_v * పరిమాణం(దశాంశం), v->nb[2], S_v * పరిమాణం(దశాంశం), chunk_కొలత);
}

స్థిర అంతఃవరుస శూన్య gdn_dma_నెట్టు_chunk_gb(
    dma_వరుస * dma_q,
    దశాంశం * vtcm_g_raw,
    దశాంశం * vtcm_b_raw,
    స్థిరరాశి నిర్మాణం htp_tensor * g,
    స్థిరరాశి నిర్మాణం htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_కొలత,
    uint32_t n_batch
) {
    స్థిరరాశి dma_addr_t g_dma    = g->దత్తాంశం + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    స్థిరరాశి dma_addr_t beta_dma = beta->దత్తాంశం + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    స్థిరరాశి uint32_t అడ్డువరుస_బైట్లు  = n_batch * పరిమాణం(దశాంశం);

    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_g_raw, g_dma), అడ్డువరుస_బైట్లు, g->nb[2], అడ్డువరుస_బైట్లు, chunk_కొలత);
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_b_raw, beta_dma), అడ్డువరుస_బైట్లు, beta->nb[2], అడ్డువరుస_బైట్లు, chunk_కొలత);
}

స్థిర అంతఃవరుస శూన్య gdn_pack_s_నిలువు_tiles(
    __fp16 * పరిమిత vtcm_s_నిలువు_tiles,
    __fp16 * పరిమిత vtcm_s_f16,
    స్థిరరాశి దశాంశం * పరిమిత vtcm_s_state,
    uint32_t S_v
) {
    కొరకు (uint32_t j = 0; j < S_v; ++j) {
        కొరకు (uint32_t i = 0; i < S_v; i += 64) {
            HVX_సదిశ v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_సదిశ v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_గమ్యం_f16(v0, v1);
        }
    }
    hmx_interleave_అడ్డువరుసలు_గమ్యం_tiles(vtcm_s_నిలువు_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);
}

నిర్మాణం htp_gdn_తల_ptrs {
    దశాంశం *  s_state;
    __fp16 * s_f16;
    __fp16 * s_నిలువు_tiles;
    దశాంశం *  s_update_f32;
    __fp16 * s_update_tiles;

    దశాంశం * q_f32[2];
    దశాంశం * k_f32[2];
    దశాంశం * v_f32[2];
    దశాంశం * g_f32[2];
    దశాంశం * b_f32[2];
    దశాంశం * o_f32[2];

    దశాంశం * v_inter_f32;
    దశాంశం * o_inter_f32;
    దశాంశం * o_intra_f32;

    __fp16 * k_f16;
    __fp16 * v_ప్రధానసంఖ్య_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_అడ్డువరుస_tiles;
    __fp16 * q_ప్రధానసంఖ్య_అడ్డువరుస_tiles;
    __fp16 * k_అడ్డువరుస_tiles;
    __fp16 * k_నిలువు_tiles;
    __fp16 * k_ప్రధానసంఖ్య_అడ్డువరుస_tiles;
    __fp16 * k_నిలువు_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_అడ్డువరుస_tiles;
    __fp16 * a_అడ్డువరుస_tiles;
    __fp16 * v_ప్రధానసంఖ్య_నిలువు_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_నిలువు_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_అడ్డువరుస_tiles;

    __fp16 * gamma;
    దశాంశం *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_సదిశ * అడ్డువరుసలు_kk;
    HVX_సదిశ * అడ్డువరుసలు_qk;
    HVX_సదిశ * అడ్డువరుసలు_inv;
    HVX_సదిశ * అడ్డువరుసలు_a;

    HVX_సదిశ * vtcm_m;
    HVX_సదిశ * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_లోపల_dma;
    dma_addr_t state_బహిర్గతం_dma;
};

స్థిర అంతఃవరుస శూన్య gdn_init_తల_ptrs(
    నిర్మాణం htp_gdn_తల_ptrs * తల,
    స్థిరరాశి నిర్మాణం htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_పునాది,
    uint32_t h,
    uint32_t పునాది_iv1,
    uint32_t iv3,
    స్థిరరాశి నిర్మాణం htp_tensor * q,
    స్థిరరాశి నిర్మాణం htp_tensor * k,
    స్థిరరాశి నిర్మాణం htp_tensor * v,
    స్థిరరాశి నిర్మాణం htp_tensor * state,
    స్థిరరాశి నిర్మాణం htp_tensor * dst,
    స్థిరరాశి నిర్మాణం htp_tensor * dst_cache,
    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_టోకెన్లు,
    uint32_t chunk_కొలత
) {
    స్థిరరాశి size_t dma_scalar_sz = hex_విడత_పైకి(chunk_కొలత * పరిమాణం(దశాంశం), 128);
    స్థిరరాశి size_t decay_sz      = 64 * 64 * పరిమాణం(__fp16);
    స్థిరరాశి size_t అడ్డువరుస_vecs_sz   = 64 * 128;

    తల->s_state        = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_s_state + h * L->state_f32_బైట్లు);
    తల->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_s_f16 + h * L->state_f16_బైట్లు);
    తల->s_నిలువు_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_s_నిలువు_tiles + h * L->state_tiles_బైట్లు);
    తల->s_update_f32   = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_s_update_f32 + h * L->state_f32_బైట్లు);
    తల->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_s_update_tiles + h * L->state_tiles_బైట్లు);

    తల->q_f32[0] = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_q_f32[0] + h * L->dma_chunk_బైట్లు);
    తల->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_q_f32[1] + h * L->dma_chunk_బైట్లు) : తల->q_f32[0];
    తల->k_f32[0] = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_k_f32[0] + h * L->dma_chunk_బైట్లు);
    తల->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_k_f32[1] + h * L->dma_chunk_బైట్లు) : తల->k_f32[0];
    తల->v_f32[0] = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_v_f32[0] + h * L->dma_chunk_బైట్లు);
    తల->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_v_f32[1] + h * L->dma_chunk_బైట్లు) : తల->v_f32[0];
    తల->g_f32[0] = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_g_f32[0] + h * dma_scalar_sz);
    తల->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_g_f32[1] + h * dma_scalar_sz) : తల->g_f32[0];
    తల->b_f32[0] = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_b_f32[0] + h * dma_scalar_sz);
    తల->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_b_f32[1] + h * dma_scalar_sz) : తల->b_f32[0];
    తల->o_f32[0] = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_o_f32[0] + h * L->dma_chunk_బైట్లు);
    తల->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_o_f32[1] + h * L->dma_chunk_బైట్లు) : తల->o_f32[0];

    తల->v_inter_f32 = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_v_inter_f32 + h * L->dma_chunk_బైట్లు);
    తల->o_inter_f32 = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_o_inter_f32 + h * L->dma_chunk_బైట్లు);
    తల->o_intra_f32 = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_o_intra_f32 + h * L->dma_chunk_బైట్లు);

    తల->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_k_f16 + h * L->act_f16_బైట్లు);
    తల->v_ప్రధానసంఖ్య_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_v_ప్రధానసంఖ్య_f16 + h * L->act_f16_బైట్లు);
    తల->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_delta_f16 + h * L->act_f16_బైట్లు);
    తల->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_d_f16 + h * L->act_f16_బైట్లు);

    తల->q_అడ్డువరుస_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_q_అడ్డువరుస_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->q_ప్రధానసంఖ్య_అడ్డువరుస_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_q_ప్రధానసంఖ్య_అడ్డువరుస_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->k_అడ్డువరుస_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_k_అడ్డువరుస_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->k_నిలువు_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_k_నిలువు_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->k_ప్రధానసంఖ్య_అడ్డువరుస_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_k_ప్రధానసంఖ్య_అడ్డువరుస_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->k_నిలువు_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_k_నిలువు_tiles_64x128 + h * L->tile_64x_Sv_బైట్లు);
    తల->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_kk_tiles + h * L->tile_64x64_బైట్లు);
    తల->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_qk_tiles + h * L->tile_64x64_బైట్లు);
    తల->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_v_inter_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_o_inter_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->inv_అడ్డువరుస_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_inv_అడ్డువరుస_tiles + h * L->tile_64x64_బైట్లు);
    తల->a_అడ్డువరుస_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_a_అడ్డువరుస_tiles + h * L->tile_64x64_బైట్లు);
    తల->v_ప్రధానసంఖ్య_నిలువు_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_v_ప్రధానసంఖ్య_నిలువు_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_delta_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->delta_నిలువు_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_delta_నిలువు_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_o_intra_tiles + h * L->tile_64x_Sv_బైట్లు);
    తల->d_అడ్డువరుస_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_d_అడ్డువరుస_tiles + h * L->tile_64x_Sv_బైట్లు);

    తల->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_gamma + h * dma_scalar_sz);
    తల->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_gamma + h * dma_scalar_sz + 128);
    తల->lambda_init     = VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L->off_lambda_init + h * dma_scalar_sz);
    తల->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_decay_m + h * decay_sz);
    తల->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_పునాది, L->off_decay_a + h * decay_sz);

    తల->అడ్డువరుసలు_kk  = VTCM_LAYOUT_PTR(HVX_సదిశ, vtcm_పునాది, L->off_అడ్డువరుసలు_kk + h * అడ్డువరుస_vecs_sz);
    తల->అడ్డువరుసలు_qk  = VTCM_LAYOUT_PTR(HVX_సదిశ, vtcm_పునాది, L->off_అడ్డువరుసలు_qk + h * అడ్డువరుస_vecs_sz);
    తల->అడ్డువరుసలు_inv = VTCM_LAYOUT_PTR(HVX_సదిశ, vtcm_పునాది, L->off_అడ్డువరుసలు_inv + h * అడ్డువరుస_vecs_sz);
    తల->అడ్డువరుసలు_a   = VTCM_LAYOUT_PTR(HVX_సదిశ, vtcm_పునాది, L->off_అడ్డువరుసలు_a + h * అడ్డువరుస_vecs_sz);

    తల->vtcm_m   = VTCM_LAYOUT_PTR(HVX_సదిశ, vtcm_పునాది, L->off_thread_scratch + h * (64 * 128));
    తల->vtcm_tmp = తల->vtcm_m + 32;

    తల->iv1 = పునాది_iv1 + h;
    తల->iv3 = iv3;
    తల->iq1 = fastmodulo(తల->iv1, q->ne[1], &kparams->div_q1);
    తల->ik1 = fastmodulo(తల->iv1, k->ne[1], &kparams->div_k1);
    తల->iq3 = fastdiv(తల->iv3, &kparams->div_rq3);
    తల->ik3 = fastdiv(తల->iv3, &kparams->div_rk3);

    తల->state_లోపల_dma = state->దత్తాంశం +
        ((uint64_t) తల->iv3 * kparams->state_seq_stride + (uint64_t) తల->iv1 * S_v * S_v) * పరిమాణం(దశాంశం);

    తల->state_బహిర్గతం_dma = dst_cache ?
        (dst_cache->దత్తాంశం + ((uint64_t) తల->iv3 * H + తల->iv1) * S_v * S_v * పరిమాణం(దశాంశం)) :
        (dst->దత్తాంశం + ((uint64_t) S_v * H * n_టోకెన్లు * kparams->n_seqs + (uint64_t) (తల->iv3 * H + తల->iv1) * S_v * S_v) * పరిమాణం(దశాంశం));
}

నిర్మాణం htp_gdn_batch_context {
    నిర్మాణం htp_gdn_తల_ptrs * heads;
    స్థిరరాశి దశాంశం *              vtcm_g_raw;
    స్థిరరాశి దశాంశం *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    దశాంశం                      scale;
    నిర్మాణం htp_ops_context *   octx;
    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams;
};

స్థిర శూన్య gdn_hvx_init_state_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    gdn_pack_s_నిలువు_tiles(తల->s_నిలువు_tiles, తల->s_f16, తల->s_state, bctx->S_v);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

స్థిర అంతఃవరుస __attribute__((వాడని_2)) HVX_సదిశ hvx_clamp_neg20_0(HVX_సదిశ v, HVX_సదిశ v_zero, HVX_సదిశ v_neg20) {
    HVX_సదిశ_Pred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_సదిశ_Pred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    తిరిగి Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_ఉపసర్గ_పరిశీలన_f32(HVX_సదిశ v, HVX_సదిశ వాహకం_లోపల) {
    స్థిరరాశి HVX_సదిశ zero = Q6_V_vzero();

    v = hvx_vec_కూడు_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_కూడు_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_కూడు_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_కూడు_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_కూడు_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_కూడు_f32_f32(v, వాహకం_లోపల);

    తిరిగి v;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_splat_చివరి_f32(HVX_సదిశ v) {
    తిరిగి hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

స్థిర శూన్య gdn_hvx_phase1a_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t curr_buf = bctx->curr_buf;
    స్థిరరాశి uint32_t S_v = bctx->S_v;
    స్థిరరాశి uint32_t n_batch = bctx->n_batch;

    ఒకవేళ (n_batch == 1) {
        hvx_vmem(తల->g_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(తల->g_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(తల->b_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(తల->b_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } లేకపోతే {
        int32_t offsets[32] __attribute__((aligned(128)));
        కొరకు (పూర్ణాంకం k = 0; k < 32; ++k) {
            offsets[k] = k * n_batch * పరిమాణం(దశాంశం);
        }
        HVX_సదిశ vv = *(స్థిరరాశి HVX_సదిశ *) offsets;
        స్థిరరాశి size_t rt_g = (size_t) ((స్థిరరాశి uint8_t *) bctx->vtcm_g_raw + i * పరిమాణం(దశాంశం));
        స్థిరరాశి size_t rt_b = (size_t) ((స్థిరరాశి uint8_t *) bctx->vtcm_b_raw + i * పరిమాణం(దశాంశం));
        స్థిరరాశి size_t mu   = 64 * n_batch * పరిమాణం(దశాంశం);

        Q6_vgather_ARMVw((HVX_సదిశ *) (తల->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_సదిశ *) (తల->g_f32[curr_buf] + 32), rt_g + 32 * n_batch * పరిమాణం(దశాంశం), mu, vv);
        Q6_vgather_ARMVw((HVX_సదిశ *) (తల->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_సదిశ *) (తల->b_f32[curr_buf] + 32), rt_b + 32 * n_batch * పరిమాణం(దశాంశం), mu, vv);
    }

    స్థిరరాశి uint32_t t_chunk = bctx->c * 64;
    స్థిరరాశి uint32_t valid_టోకెన్లు = hex_smin(64, bctx->kparams->n_టోకెన్లు - t_chunk);
    ఒకవేళ (valid_టోకెన్లు < 64) {
        కొరకు (uint32_t t = valid_టోకెన్లు; t < 64; ++t) {
            తల->g_f32[curr_buf][t] = 0.0f;
            తల->b_f32[curr_buf][t] = 0.0f;
        }
        స్థిరరాశి HVX_సదిశ vzero = Q6_V_vzero();
        కొరకు (uint32_t t = valid_టోకెన్లు; t < 64; ++t) {
            కొరకు (uint32_t j = 0; j < S_v; j += 32) {
                hvx_vmem(తల->q_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(తల->k_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(తల->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    స్థిరరాశి HVX_సదిశ v_g0 = hvx_vmem(తల->g_f32[curr_buf] + 0);
    స్థిరరాశి HVX_సదిశ v_g1 = hvx_vmem(తల->g_f32[curr_buf] + 32);

    HVX_సదిశ v_gamma0 = hvx_ఉపసర్గ_పరిశీలన_f32(v_g0, Q6_V_vzero());
    HVX_సదిశ v_వాహకం  = hvx_splat_చివరి_f32(v_gamma0);
    HVX_సదిశ v_gamma1 = hvx_ఉపసర్గ_పరిశీలన_f32(v_g1, v_వాహకం);

    స్థిరరాశి HVX_సదిశ v_zero  = Q6_V_vzero();
    స్థిరరాశి HVX_సదిశ v_neg20 = hvx_vec_splat_f32(-20.0f);

    hvx_vmem(తల->gamma) = hvx_vec_f32_గమ్యం_f16(v_gamma0, v_gamma1);

    HVX_సదిశ v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_సదిశ v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    hvx_vmem(తల->lambda_init + 0)  = v_l0;
    hvx_vmem(తల->lambda_init + 32) = v_l1;
    hvx_vmem(తల->lambda_init_f16)  = hvx_vec_f32_గమ్యం_f16(v_l0, v_l1);

    gdn_f32_గమ్యం_hmx_అడ్డువరుస_tiles_and_f16(తల->k_అడ్డువరుస_tiles, తల->k_ప్రధానసంఖ్య_అడ్డువరుస_tiles, తల->k_f16,
                                     తల->k_f32[curr_buf], తల->lambda_init_f16, 64, S_v);
    hmx_interleave_అడ్డువరుసలు_గమ్యం_tiles(తల->k_నిలువు_tiles, తల->k_f16, 64, S_v, S_v, 0, 64);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

స్థిర శూన్య gdn_hvx_phase1b_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t curr_buf = bctx->curr_buf;
    స్థిరరాశి uint32_t S_v = bctx->S_v;

    gdn_f32_గమ్యం_hmx_అడ్డువరుస_tiles_and_f16(తల->q_అడ్డువరుస_tiles, తల->q_ప్రధానసంఖ్య_అడ్డువరుస_tiles, ఖాళీ,
                                     తల->q_f32[curr_buf], తల->lambda_init_f16, 64, S_v);

    hmx_interleave_నిలువుల_సంఖ్య_గమ్యం_tiles(తల->k_నిలువు_tiles_64x128, తల->k_f16, 64, S_v, S_v, 2, 0, 64);

    స్థిరరాశి uint16_t * gamma_u16 = (స్థిరరాశి uint16_t *) తల->gamma;
    స్థిరరాశి HVX_సదిశ v_gamma   = hvx_vmem(తల->gamma);

    స్థిరరాశి HVX_సదిశ v_zero_f16  = Q6_V_vzero();
    స్థిరరాశి HVX_సదిశ v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    స్థిరరాశి HVX_సదిశ v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    స్థిరరాశి HVX_సదిశ v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(తల->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(తల->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    కొరకు (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_సదిశ v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_సదిశ v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_సదిశ diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_సదిశ diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_సదిశ_Pred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_సదిశ_Pred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_సదిశ diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_సదిశ diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_సదిశ v_exp0      = hvx_vec_exp2_f16(diff_log2e0);
        HVX_సదిశ v_exp1      = hvx_vec_exp2_f16(diff_log2e1);

        HVX_సదిశ_Pred ముసుగు_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_సదిశ_Pred ముసుగు_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_సదిశ v_m0         = Q6_V_vand_QV(ముసుగు_lt0, v_exp0);
        HVX_సదిశ v_m1         = Q6_V_vand_QV(ముసుగు_lt1, v_exp1);

        HVX_సదిశ_Pred ముసుగు_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_సదిశ_Pred ముసుగు_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_సదిశ_Pred ముసుగు_diag0 = Q6_Q_and_QQn(ముసుగు_le0, ముసుగు_lt0);
        HVX_సదిశ_Pred ముసుగు_diag1 = Q6_Q_and_QQn(ముసుగు_le1, ముసుగు_lt1);

        HVX_సదిశ v_a0         = Q6_V_vmux_QVV(ముసుగు_diag0, v_one_f16, v_m0);
        HVX_సదిశ v_a1         = Q6_V_vmux_QVV(ముసుగు_diag1, v_one_f16, v_m1);

        hvx_vmem(తల->decay_m + t0 * 64) = v_m0;
        hvx_vmem(తల->decay_a + t0 * 64) = v_a0;
        hvx_vmem(తల->decay_m + t1 * 64) = v_m1;
        hvx_vmem(తల->decay_a + t1 * 64) = v_a1;
    }

    {
        HVX_సదిశ v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_సదిశ diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_సదిశ_Pred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_సదిశ diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_సదిశ v_exp         = hvx_vec_exp2_f16(diff_log2e);

        HVX_సదిశ_Pred ముసుగు_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_సదిశ v_m           = Q6_V_vand_QV(ముసుగు_lt_t, v_exp);

        HVX_సదిశ_Pred ముసుగు_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_సదిశ_Pred ముసుగు_diag = Q6_Q_and_QQn(ముసుగు_le_t, ముసుగు_lt_t);
        HVX_సదిశ v_a           = Q6_V_vmux_QVV(ముసుగు_diag, v_one_f16, v_m);

        hvx_vmem(తల->decay_m + 63 * 64) = v_m;
        hvx_vmem(తల->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

స్థిర శూన్య gdn_hvx_phase2_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t curr_buf = bctx->curr_buf;

    gdn_unpack_64x64_tiles_గమ్యం_vectors(తల->అడ్డువరుసలు_kk, తల->kk_tiles);

    gdn_build_inv_l_ఖండాలు(
        తల->inv_అడ్డువరుస_tiles,
        తల->అడ్డువరుసలు_kk,
        తల->decay_m,
        తల->b_f32[curr_buf],
        (__fp16 *) తల->vtcm_m,
        (__fp16 *) తల->vtcm_m + HMX_FP16_TILE_N_ELMS
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

స్థిర శూన్య gdn_hvx_phase3_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t curr_buf = bctx->curr_buf;
    స్థిరరాశి uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_గమ్యం_f32(తల->v_inter_f32, తల->v_inter_tiles, S_v);

    HVX_సదిశ_Alias local_b[2];
    local_b[0].v = hvx_vmem(తల->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(తల->b_f32[curr_buf] + 32);

    కొరకు (uint32_t t = 0; t < 64; ++t) {
        HVX_సదిశ vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
        కొరకు (uint32_t j = 0; j < S_v; j += 64) {
            HVX_సదిశ vv0 = hvx_vmem(తల->v_f32[curr_buf] + t * S_v + j + 0);
            HVX_సదిశ vv1 = (j + 32 < S_v) ? hvx_vmem(తల->v_f32[curr_buf] + t * S_v + j + 32) : Q6_V_vzero();
            HVX_సదిశ vi0 = hvx_vmem(తల->v_inter_f32 + t * S_v + j + 0);
            HVX_సదిశ vi1 = (j + 32 < S_v) ? hvx_vmem(తల->v_inter_f32 + t * S_v + j + 32) : Q6_V_vzero();

            HVX_సదిశ vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
            HVX_సదిశ vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

            hvx_vmem(తల->v_ప్రధానసంఖ్య_f16 + t * S_v + j) = hvx_vec_f32_గమ్యం_f16(vp0, vp1);
        }
    }

    hmx_interleave_నిలువుల_సంఖ్య_గమ్యం_tiles(తల->v_ప్రధానసంఖ్య_నిలువు_tiles, తల->v_ప్రధానసంఖ్య_f16, 64, S_v, S_v, 2, 0, 64);

    gdn_unpack_64x64_tiles_గమ్యం_vectors(తల->అడ్డువరుసలు_qk, తల->qk_tiles);
    కొరకు (uint32_t t = 0; t < 64; ++t) {
        HVX_సదిశ v_decay_a = hvx_vmem(తల->decay_a + t * 64);
        తల->అడ్డువరుసలు_a[t]      = hvx_vec_mul_f16_f16(తల->అడ్డువరుసలు_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_గమ్యం_tiles(తల->a_అడ్డువరుస_tiles, తల->అడ్డువరుసలు_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

స్థిర శూన్య gdn_hvx_phase4_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_గమ్యం_f16(తల->delta_f16, తల->delta_tiles, S_v);
    hmx_interleave_నిలువుల_సంఖ్య_గమ్యం_tiles(తల->delta_నిలువు_tiles, తల->delta_f16, 64, S_v, S_v, 2, 0, 64);

    స్థిరరాశి uint16_t * decay_చివరి = (స్థిరరాశి uint16_t *) (తల->decay_a + 63 * 64);
    స్థిరరాశి HVX_సదిశ vzero      = Q6_V_vzero();

    కొరకు (uint32_t s = 0; s < 64; ++s) {
        HVX_సదిశ vs         = Q6_Vh_vsplat_R(decay_చివరి[s]);
        HVX_సదిశ_Pred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
        కొరకు (uint32_t j = 0; j < S_v; j += 64) {
            HVX_సదిశ vd   = hvx_vmem(తల->delta_f16 + s * S_v + j);
            HVX_సదిశ prod = hvx_vec_mul_f16_f16(vd, vs);
            hvx_vmem(తల->d_f16 + s * S_v + j) = Q6_V_vmux_QVV(p_zero, vzero, prod);
        }
    }

    gdn_pack_d_t_అడ్డువరుస_tiles(తల->d_అడ్డువరుస_tiles, తల->d_f16, S_v, తల->vtcm_m, తల->vtcm_tmp);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

స్థిర శూన్య gdn_hvx_phase5_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_బహిర్గతం, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t curr_buf = bctx->curr_buf;
    స్థిరరాశి uint32_t S_v = bctx->S_v;
    స్థిరరాశి దశాంశం scale = bctx->scale;

    gdn_unpack_64x_S_tiles_గమ్యం_f32(తల->o_inter_f32, తల->o_inter_tiles, S_v);
    gdn_unpack_64x_S_tiles_గమ్యం_f32(తల->o_intra_f32, తల->o_intra_tiles, S_v);

    HVX_సదిశ vscale = hvx_vec_splat_f32(scale);
    కొరకు (uint32_t j = 0; j < 64 * S_v / 32; ++j) {
        HVX_సదిశ vi = hvx_vmem(తల->o_inter_f32 + j * 32);
        HVX_సదిశ va = hvx_vmem(తల->o_intra_f32 + j * 32);
        hvx_vmem(తల->o_f32[curr_buf] + j * 32) = hvx_vec_mul_f32_f32(hvx_vec_కూడు_f32_f32(vi, va), vscale);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_బహిర్గతం, info);
}

స్థిర శూన్య gdn_hvx_phase6_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    (శూన్య) n;
    నిర్మాణం htp_gdn_batch_context * bctx = (నిర్మాణం htp_gdn_batch_context *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    స్థిరరాశి uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    నిర్మాణం htp_gdn_తల_ptrs * తల = &bctx->heads[i];
    స్థిరరాశి uint32_t S_v = bctx->S_v;
    స్థిరరాశి uint32_t c = bctx->c;
    స్థిరరాశి uint32_t n_chunks = bctx->kparams->n_chunks;

    gdn_unpack_Sx_S_tiles_గమ్యం_f32(తల->s_update_f32, తల->s_update_tiles, S_v);

    HVX_సదిశ_Alias చివరి_lambda;
    చివరి_lambda.v = hvx_vmem(తల->lambda_init + 32);
    HVX_సదిశ v_l_final = hvx_vec_splat_f32(చివరి_lambda.fp32[31]);

    కొరకు (uint32_t j = 0; j < S_v * S_v / 32; ++j) {
        HVX_సదిశ vs_old = hvx_vmem(తల->s_state + j * 32);
        HVX_సదిశ vsu    = hvx_vmem(తల->s_update_f32 + j * 32);
        hvx_vmem(తల->s_state + j * 32) = hvx_vec_కూడు_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
    }

    ఒకవేళ (c + 1 < n_chunks) {
        gdn_pack_s_నిలువు_tiles(తల->s_నిలువు_tiles, తల->s_f16, తల->s_state, S_v);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


స్థిర పూర్ణాంకం gated_delta_net_f32_hmx_chunked(
    నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams,
    uint32_t అడ్డువరుస_ప్రారంభం,
    uint32_t nrows
) {
    స్థిరరాశి నిర్మాణం htp_tensor * q         = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k         = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v         = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * g         = octx->src[3];
    స్థిరరాశి నిర్మాణం htp_tensor * beta      = octx->src[4];
    స్థిరరాశి నిర్మాణం htp_tensor * state     = octx->src[5];
    స్థిరరాశి నిర్మాణం htp_tensor * dst       = octx->dst;
    స్థిరరాశి నిర్మాణం htp_tensor * dst_cache = octx->dsts[1];

    స్థిరరాశి uint32_t S_v        = kparams->S_v;
    స్థిరరాశి uint32_t H          = kparams->H;
    స్థిరరాశి uint32_t n_టోకెన్లు   = kparams->n_టోకెన్లు;
    స్థిరరాశి దశాంశం    scale      = kparams->scale;
    స్థిరరాశి uint32_t chunk_కొలత = kparams->chunk_కొలత;
    స్థిరరాశి uint32_t n_chunks   = kparams->n_chunks;
    స్థిరరాశి uint32_t n_sv_tiles = S_v / 32;

    నిర్మాణం htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_కొలత, kparams->n_heads_batch, kparams->n_దారాలు, kparams->pipeline != 0);

    ఒకవేళ (L.మొత్తం_బైట్లు > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి vtcm_పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;

    దశాంశం * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L.off_g_raw[0])
    };
    దశాంశం * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(దశాంశం, vtcm_పునాది, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_పునాది, L.off_scales_1);
    hmx_init_స్తంభం_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_వరుస_t  hmx_q = octx->ctx->hmx_వరుస;
    dma_వరుస *  dma_q = octx->ctx->dma[0];
    work_వరుస_t wp    = octx->ctx->work_వరుస;

    నిర్మాణం htp_gdn_తల_ptrs heads[8];
    నిర్మాణం htp_gdn_hmx_gemm_task gemm_tasks[8][9];

    uint32_t n_batch = 1;
    కొరకు (uint32_t r = అడ్డువరుస_ప్రారంభం; r < అడ్డువరుస_ప్రారంభం + nrows; r += n_batch) {
        స్థిరరాశి uint32_t తల_లోపల_seq         = fastmodulo(r, H, &kparams->div_H);
        స్థిరరాశి uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        స్థిరరాశి uint32_t heads_ఎడమ_లోపల_seq   = H - తల_లోపల_seq;
        స్థిరరాశి uint32_t heads_ఎడమ_లోపల_range = (అడ్డువరుస_ప్రారంభం + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_ఎడమ_లోపల_seq, heads_ఎడమ_లోపల_range));

        కొరకు (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_తల_ptrs(&heads[h], &L, vtcm_పునాది, h, తల_లోపల_seq, iv3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_టోకెన్లు, chunk_కొలత);
        }

        నిర్మాణం htp_gdn_batch_context bctx;
        bctx.heads      = heads;
        bctx.vtcm_g_raw = ఖాళీ;
        bctx.vtcm_b_raw = ఖాళీ;
        bctx.curr_buf   = 0;
        bctx.c          = 0;
        bctx.n_batch    = n_batch;
        bctx.S_v        = S_v;
        bctx.scale      = scale;
        bctx.octx       = octx;
        bctx.kparams    = kparams;

        కొరకు (uint32_t h = 0; h < n_batch; ++h) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(heads[h].s_state, heads[h].state_లోపల_dma),
                           S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం), S_v);
        }
        కొరకు (uint32_t h = 0; h < n_batch; ++h) {
            dma_వరుస_తీయి(dma_q);
        }

        ఒకవేళ (n_chunks > 0) {
            work_వరుస_నిరంతరం(wp, gdn_hvx_init_state_కార్మికుడు, &bctx, n_batch);

            స్థిరరాశి uint32_t chunk0_టోకెన్లు = hex_smin(chunk_కొలత, n_టోకెన్లు);
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                gdn_dma_నెట్టు_chunk_inputs(dma_q, heads[h].q_f32[0], heads[h].k_f32[0], heads[h].v_f32[0],
                                          q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                          heads[h].iv3, heads[h].iv1, 0, chunk0_టోకెన్లు, S_v);
            }
            gdn_dma_నెట్టు_chunk_gb(dma_q, vtcm_g_raw[0], vtcm_b_raw[0], g, beta, iv3, తల_లోపల_seq, 0, chunk0_టోకెన్లు, n_batch);
        }

        కొరకు (uint32_t c = 0; c < n_chunks; ++c) {
            స్థిరరాశి uint32_t curr_buf = c & 1;
            స్థిరరాశి uint32_t తదుపరి_buf = (c + 1) & 1;
            స్థిరరాశి uint32_t t_chunk  = c * chunk_కొలత;

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                dma_వరుస_తీయి(dma_q);
                dma_వరుస_తీయి(dma_q);
                dma_వరుస_తీయి(dma_q);
            }
            dma_వరుస_తీయి(dma_q);
            dma_వరుస_తీయి(dma_q);

            ఒకవేళ (c + 1 < n_chunks) {
                స్థిరరాశి uint32_t తదుపరి_t_chunk = (c + 1) * chunk_కొలత;
                స్థిరరాశి uint32_t తదుపరి_టోకెన్లు_2  = hex_smin(chunk_కొలత, n_టోకెన్లు - తదుపరి_t_chunk);
                కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_నెట్టు_chunk_inputs(dma_q, heads[h].q_f32[తదుపరి_buf], heads[h].k_f32[తదుపరి_buf], heads[h].v_f32[తదుపరి_buf],
                                              q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                              heads[h].iv3, heads[h].iv1, తదుపరి_t_chunk, తదుపరి_టోకెన్లు_2, S_v);
                }
                gdn_dma_నెట్టు_chunk_gb(dma_q, vtcm_g_raw[తదుపరి_buf], vtcm_b_raw[తదుపరి_buf],
                                      g, beta, iv3, తల_లోపల_seq, తదుపరి_t_chunk, తదుపరి_టోకెన్లు_2, n_batch);
            }

            ఒకవేళ (c > 0) {
                కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                    dma_వరుస_తీయి(dma_q);
                }
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase1a_కార్మికుడు, &bctx, n_batch);

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][0], heads[h].k_అడ్డువరుస_tiles, heads[h].k_నిలువు_tiles, heads[h].kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][2], heads[h].k_ప్రధానసంఖ్య_అడ్డువరుస_tiles, heads[h].s_నిలువు_tiles, heads[h].v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase1b_కార్మికుడు, &bctx, n_batch);

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][1], heads[h].q_అడ్డువరుస_tiles, heads[h].k_నిలువు_tiles, heads[h].qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][3], heads[h].q_ప్రధానసంఖ్య_అడ్డువరుస_tiles, heads[h].s_నిలువు_tiles, heads[h].o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase2_కార్మికుడు, &bctx, n_batch);

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][7],
                    (__fp16 *) heads[h].vtcm_m,
                    heads[h].inv_అడ్డువరుస_tiles + 0 * HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    1, 1, 1, vtcm_scales_1
                );
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][8],
                    (__fp16 *) heads[h].vtcm_m + HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    heads[h].inv_అడ్డువరుస_tiles + 2 * HMX_FP16_TILE_N_ELMS,
                    1, 1, 1, vtcm_scales_1
                );
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase3_కార్మికుడు, &bctx, n_batch);

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][4], heads[h].inv_అడ్డువరుస_tiles, heads[h].v_ప్రధానసంఖ్య_నిలువు_tiles, heads[h].delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase4_కార్మికుడు, &bctx, n_batch);

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][5], heads[h].a_అడ్డువరుస_tiles, heads[h].delta_నిలువు_tiles, heads[h].o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_నెట్టు_hmx_gemm_task(hmx_q, &gemm_tasks[h][6], heads[h].d_అడ్డువరుస_tiles, heads[h].k_నిలువు_tiles_64x128, heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);
            }

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase5_కార్మికుడు, &bctx, n_batch);

            స్థిరరాశి uint32_t valid_టోకెన్లు = hex_smin(chunk_కొలత, n_టోకెన్లు - t_chunk);
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                స్థిరరాశి dma_addr_t attn_chunk_dma = dst->దత్తాంశం +
                    ((uint64_t) heads[h].iv3 * n_టోకెన్లు * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * పరిమాణం(దశాంశం);
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం), valid_టోకెన్లు);
            }

            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                hmx_వరుస_తీయి(hmx_q);
            }

            work_వరుస_నిరంతరం(wp, gdn_hvx_phase6_కార్మికుడు, &bctx, n_batch);
        }

        ఒకవేళ (n_chunks > 0) {
            కొరకు (uint32_t h = 0; h < n_batch; ++h) {
                dma_వరుస_తీయి(dma_q);
            }
        }

        కొరకు (uint32_t h = 0; h < n_batch; ++h) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(heads[h].state_బహిర్గతం_dma, heads[h].s_state),
                           S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం), S_v * పరిమాణం(దశాంశం), S_v);
        }
        కొరకు (uint32_t h = 0; h < n_batch; ++h) {
            dma_వరుస_తీయి(dma_q);
        }
    }

    dma_వరుస_flush(dma_q);
    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_gated_delta_net(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * q     = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k     = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v     = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * g     = octx->src[3];
    స్థిరరాశి నిర్మాణం htp_tensor * beta  = octx->src[4];
    స్థిరరాశి నిర్మాణం htp_tensor * state = octx->src[5];
    స్థిరరాశి నిర్మాణం htp_tensor * dst   = octx->dst;

    ఒకవేళ (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t S_v      = v->ne[0];
    స్థిరరాశి uint32_t H        = v->ne[1];
    స్థిరరాశి uint32_t n_టోకెన్లు = v->ne[2];
    స్థిరరాశి uint32_t n_seqs   = v->ne[3];
    స్థిరరాశి uint32_t K        = octx->op_params[0];

    ఒకవేళ (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_టోకెన్లు == 0 || n_seqs == 0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    ఒకవేళ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    ఒకవేళ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_టోకెన్లు || k->ne[2] != n_టోకెన్లు || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    ఒకవేళ (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    ఒకవేళ (dst->ne[0] != S_v * H || dst->ne[1] != n_టోకెన్లు * n_seqs + S_v * n_seqs * K) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 5; i++) {
        ఒకవేళ (htp_tensor_is_extended(octx->src[i])) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    }
    ఒకవేళ (htp_tensor_is_extended(octx->dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    ఒకవేళ (octx->dsts[1]) {
        స్థిరరాశి నిర్మాణం htp_tensor * dst_cache = octx->dsts[1];
        ఒకవేళ (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    }

    స్థిరరాశి నిర్మాణం htp_gdn_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_gdn_kernel_params *) octx->kernel_params;
    నిర్మాణం htp_gdn_kernel_params kparams_local;
    ఒకవేళ (!kparams || kparams->S_v == 0) {
        స్థిరరాశి uint32_t rq3 = n_seqs / q->ne[3];
        స్థిరరాశి uint32_t rk3 = n_seqs / k->ne[3];
        స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = H * n_seqs;
        uint32_t n_దారాలు = (మొత్తం_అడ్డువరుసలు < octx->n_దారాలు) ? మొత్తం_అడ్డువరుసలు : octx->n_దారాలు;
        ఒకవేళ (n_దారాలు == 0) {
            n_దారాలు = 1;
        }

        memset(&kparams_local, 0, పరిమాణం(kparams_local));
        kparams_local.n_దారాలు           = n_దారాలు;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_టోకెన్లు            = n_టోకెన్లు;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.మొత్తం_అడ్డువరుసలు          = మొత్తం_అడ్డువరుసలు;
        kparams_local.అడ్డువరుసలు_చొప్పున_thread     = (మొత్తం_అడ్డువరుసలు + n_దారాలు - 1) / n_దారాలు;
        స్థిరరాశి bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_టోకెన్లు >= HTP_GDN_MIN_టోకెన్లు) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        నిర్మాణం htp_gdn_hmx_vtcm_layout hmx_layout_local;
        నిర్మాణం htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        ఒకవేళ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_కొలత, మొత్తం_అడ్డువరుసలు, octx->ctx->vtcm_కొలత, n_దారాలు, true, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_కొలత      = HTP_GDN_CHUNK_కొలత;
            kparams_local.n_chunks        = (n_టోకెన్లు + HTP_GDN_CHUNK_కొలత - 1) / HTP_GDN_CHUNK_కొలత;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_కొలత       = (uint32_t) hmx_layout_local.మొత్తం_బైట్లు;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_బైట్లు;
            kparams_local.vtcm_చొప్పున_thread = (uint32_t) (hmx_layout_local.మొత్తం_బైట్లు / (n_దారాలు > 0 ? n_దారాలు : 1));
        } లేకపోతే {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_దారాలు);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_చొప్పున_thread = (uint32_t) hvx_layout_local.బైట్లు_చొప్పున_thread;
            kparams_local.vtcm_కొలత       = (uint32_t) hvx_layout_local.మొత్తం_బైట్లు;
        }
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((దశాంశం) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / పరిమాణం(దశాంశం));
        kparams_local.state_కొలత_చొప్పున_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_విలువలు(H);
        kparams_local.div_q1        = init_fastdiv_విలువలు(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_విలువలు(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_విలువలు(rq3);
        kparams_local.div_rk3       = init_fastdiv_విలువలు(rk3);
        kparams_local.div_n_దారాలు = init_fastdiv_విలువలు(n_దారాలు);

        kparams = &kparams_local;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = kparams->మొత్తం_అడ్డువరుసలు;
    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి bool can_విభజించు = htp_tensor_mdev_దత్తాంశం_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            మొత్తం_అడ్డువరుసలు,
            can_విభజించు ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.లెక్క,
            &octx->ctx->mdev.లెక్క_div
        );
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    } లేకపోతే ఒకవేళ (octx->op_params[1] != 0) {
        అడ్డువరుస_ప్రారంభం = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        తిరిగి gated_delta_net_f32_hmx_chunked(octx, kparams, అడ్డువరుస_ప్రారంభం, nrows);
    }

    స్థిరరాశి uint32_t n_దారాలు = (nrows < kparams->n_దారాలు) ? nrows : kparams->n_దారాలు;

    నిర్మాణం htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.అడ్డువరుస_ప్రారంభం = అడ్డువరుస_ప్రారంభం;
    gctx.nrows     = nrows;
    gctx.vtcm_పునాది = octx->ctx->vtcm_పునాది;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_దారాలు);

    ఒకవేళ (gctx.layout.మొత్తం_బైట్లు > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    FARF(ఎగువ_2, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.మొత్తం_బైట్లు, n_దారాలు);

    ఒకవేళ (n_టోకెన్లు == 1) {
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, gated_delta_net_f32_tg_thread, &gctx, n_దారాలు);
    } లేకపోతే {
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, gated_delta_net_f32_pp_thread, &gctx, n_దారాలు);
    }

    తిరిగి HTP_స్థితి_OK;
}
