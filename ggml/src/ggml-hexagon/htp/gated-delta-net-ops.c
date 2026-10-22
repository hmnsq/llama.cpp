#شامل <stdint.h>
#شامل <stdbool.h>
#شامل <string.h>
#شامل <math.h>
#شامل <HAP_farf.h>

#شامل "hvx-base.h"
#شامل "hvx-copy.h"
#شامل "hvx-reduce.h"
#شامل "hvx-exp.h"
#شامل "dma-queue.h"
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-tensor.h"
#شامل "htp-vtcm.h"
#شامل "hmx-utils.h"
#شامل "hmx-fa-kernels.h"
#شامل "hmx-queue.h"
#شامل "gated-delta-net-ops.h"

#اگر_غیر_تعریف_شدہ MIN
#تعریف_کرو MIN(a, b) ((a) < (b) ? (a) : (b))
#اگر_ختم

ساخت htp_gdn_context {
    ساخت htp_ops_context * octx;
    ثابت ساخت htp_gdn_kernel_params * kparams;
    ساخت htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_بنیاد;
    uint32_t صف_آغاز;
    uint32_t nrows;
};

ساکن اندرونی HVX_سمتیہ gdn_mul_dot_f32(اعشاری * محدود dst, ثابت HVX_سمتیہ * محدود mul, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc = Q6_V_vzero();
    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vd   = hvx_vmemu(dst + i * epv);
        HVX_سمتیہ vm   = mul[i];
        HVX_سمتیہ vdot = dot[i];
        HVX_سمتیہ برآمد  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = برآمد;
        acc = hvx_vec_جوڑو_f32_f32(acc, hvx_vec_mul_f32_f32(برآمد, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vm = mul[nvec];
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ برآمد = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * حجم(اعشاری), برآمد);
        acc = hvx_vec_جوڑو_f32_f32(acc, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(برآمد, vdot), zero));
    }

    واپس hvx_vec_reduce_مجموع_f32(acc);
}

ساکن اندرونی HVX_سمتیہ gdn_mul_scalar_dot_f32(اعشاری * محدود dst, HVX_سمتیہ vmul, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc = Q6_V_vzero();
    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vd   = hvx_vmemu(dst + i * epv);
        HVX_سمتیہ vdot = dot[i];
        HVX_سمتیہ برآمد  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = برآمد;
        acc = hvx_vec_جوڑو_f32_f32(acc, hvx_vec_mul_f32_f32(برآمد, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ برآمد = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * حجم(اعشاری), برآمد);
        acc = hvx_vec_جوڑو_f32_f32(acc, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(برآمد, vdot), zero));
    }

    واپس hvx_vec_reduce_مجموع_f32(acc);
}

ساکن اندرونی HVX_سمتیہ gdn_جوڑو_scaled_dot_f32(اعشاری * محدود dst, ثابت HVX_سمتیہ * محدود src,
        HVX_سمتیہ vscale, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc = Q6_V_vzero();
    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vd   = hvx_vmemu(dst + i * epv);
        HVX_سمتیہ vs   = src[i];
        HVX_سمتیہ vdot = dot[i];
        HVX_سمتیہ برآمد  = hvx_vec_جوڑو_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = برآمد;
        acc = hvx_vec_جوڑو_f32_f32(acc, hvx_vec_mul_f32_f32(برآمد, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vs   = src[nvec];
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ برآمد = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * حجم(اعشاری), برآمد);
        acc = hvx_vec_جوڑو_f32_f32(acc, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(برآمد, vdot), zero));
    }

    واپس hvx_vec_reduce_مجموع_f32(acc);
}

ساکن اندرونی HVX_سمتیہ gdn_mul_dot4_f32(اعشاری * محدود dst0, اعشاری * محدود dst1,
        اعشاری * محدود dst2, اعشاری * محدود dst3,
        ثابت HVX_سمتیہ * محدود mul, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc0 = Q6_V_vzero();
    HVX_سمتیہ acc1 = Q6_V_vzero();
    HVX_سمتیہ acc2 = Q6_V_vzero();
    HVX_سمتیہ acc3 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vm   = mul[i];
        HVX_سمتیہ vdot = dot[i];

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vm = mul[nvec];
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * حجم(اعشاری), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(اعشاری), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(اعشاری), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(اعشاری), out3);

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_سمتیہ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    واپس hvx_vec_reduce_مجموع_f32x4(acc);
}

ساکن اندرونی HVX_سمتیہ gdn_mul_scalar_dot4_f32(اعشاری * محدود dst0, اعشاری * محدود dst1,
        اعشاری * محدود dst2, اعشاری * محدود dst3,
        HVX_سمتیہ vmul, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc0 = Q6_V_vzero();
    HVX_سمتیہ acc1 = Q6_V_vzero();
    HVX_سمتیہ acc2 = Q6_V_vzero();
    HVX_سمتیہ acc3 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vdot = dot[i];

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * حجم(اعشاری), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(اعشاری), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(اعشاری), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(اعشاری), out3);

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_سمتیہ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    واپس hvx_vec_reduce_مجموع_f32x4(acc);
}

ساکن اندرونی HVX_سمتیہ gdn_جوڑو_scaled_dot4_f32(اعشاری * محدود dst0, اعشاری * محدود dst1,
        اعشاری * محدود dst2, اعشاری * محدود dst3,
        ثابت HVX_سمتیہ * محدود src, ثابت اعشاری * محدود scale,
        ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc0 = Q6_V_vzero();
    HVX_سمتیہ acc1 = Q6_V_vzero();
    HVX_سمتیہ acc2 = Q6_V_vzero();
    HVX_سمتیہ acc3 = Q6_V_vzero();
    ثابت HVX_سمتیہ scale0 = hvx_vec_splat_f32(scale[0]);
    ثابت HVX_سمتیہ scale1 = hvx_vec_splat_f32(scale[1]);
    ثابت HVX_سمتیہ scale2 = hvx_vec_splat_f32(scale[2]);
    ثابت HVX_سمتیہ scale3 = hvx_vec_splat_f32(scale[3]);

    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vs   = src[i];
        HVX_سمتیہ vdot = dot[i];

        HVX_سمتیہ out0 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_سمتیہ out1 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_سمتیہ out2 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_سمتیہ out3 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vs   = src[nvec];
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ out0 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_سمتیہ out1 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_سمتیہ out2 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_سمتیہ out3 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * حجم(اعشاری), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(اعشاری), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(اعشاری), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(اعشاری), out3);

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_سمتیہ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    واپس hvx_vec_reduce_مجموع_f32x4(acc);
}

ساکن اندرونی HVX_سمتیہ gdn_mul_dot8_f32(اعشاری * محدود dst0, اعشاری * محدود dst1,
        اعشاری * محدود dst2, اعشاری * محدود dst3, اعشاری * محدود dst4,
        اعشاری * محدود dst5, اعشاری * محدود dst6, اعشاری * محدود dst7,
        ثابت HVX_سمتیہ * محدود mul, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc0 = Q6_V_vzero();
    HVX_سمتیہ acc1 = Q6_V_vzero();
    HVX_سمتیہ acc2 = Q6_V_vzero();
    HVX_سمتیہ acc3 = Q6_V_vzero();
    HVX_سمتیہ acc4 = Q6_V_vzero();
    HVX_سمتیہ acc5 = Q6_V_vzero();
    HVX_سمتیہ acc6 = Q6_V_vzero();
    HVX_سمتیہ acc7 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vm   = mul[i];
        HVX_سمتیہ vdot = dot[i];

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_سمتیہ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_سمتیہ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_سمتیہ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_سمتیہ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_جوڑو_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_جوڑو_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_جوڑو_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_جوڑو_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vm = mul[nvec];
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_سمتیہ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_سمتیہ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_سمتیہ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_سمتیہ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * حجم(اعشاری), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(اعشاری), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(اعشاری), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(اعشاری), out3);
        hvx_vec_store_u(dst4 + off, nloe * حجم(اعشاری), out4);
        hvx_vec_store_u(dst5 + off, nloe * حجم(اعشاری), out5);
        hvx_vec_store_u(dst6 + off, nloe * حجم(اعشاری), out6);
        hvx_vec_store_u(dst7 + off, nloe * حجم(اعشاری), out7);

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_جوڑو_f32_f32(acc4, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_جوڑو_f32_f32(acc5, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_جوڑو_f32_f32(acc6, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_جوڑو_f32_f32(acc7, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_سمتیہ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_سمتیہ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_سمتیہ rA = hvx_vec_reduce_مجموع_f32x4(accA);
    HVX_سمتیہ rB = hvx_vec_reduce_مجموع_f32x4(accB);
    HVX_سمتیہ_Pred q16 = Q6_Q_vsetq2_R(16);
    واپس Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ساکن اندرونی HVX_سمتیہ gdn_mul_scalar_dot8_f32(اعشاری * محدود dst0, اعشاری * محدود dst1,
        اعشاری * محدود dst2, اعشاری * محدود dst3, اعشاری * محدود dst4,
        اعشاری * محدود dst5, اعشاری * محدود dst6, اعشاری * محدود dst7,
        HVX_سمتیہ vmul, ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc0 = Q6_V_vzero();
    HVX_سمتیہ acc1 = Q6_V_vzero();
    HVX_سمتیہ acc2 = Q6_V_vzero();
    HVX_سمتیہ acc3 = Q6_V_vzero();
    HVX_سمتیہ acc4 = Q6_V_vzero();
    HVX_سمتیہ acc5 = Q6_V_vzero();
    HVX_سمتیہ acc6 = Q6_V_vzero();
    HVX_سمتیہ acc7 = Q6_V_vzero();

    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vdot = dot[i];

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_سمتیہ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_سمتیہ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_سمتیہ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_سمتیہ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_جوڑو_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_جوڑو_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_جوڑو_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_جوڑو_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_سمتیہ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_سمتیہ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_سمتیہ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_سمتیہ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_سمتیہ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_سمتیہ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_سمتیہ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * حجم(اعشاری), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(اعشاری), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(اعشاری), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(اعشاری), out3);
        hvx_vec_store_u(dst4 + off, nloe * حجم(اعشاری), out4);
        hvx_vec_store_u(dst5 + off, nloe * حجم(اعشاری), out5);
        hvx_vec_store_u(dst6 + off, nloe * حجم(اعشاری), out6);
        hvx_vec_store_u(dst7 + off, nloe * حجم(اعشاری), out7);

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_جوڑو_f32_f32(acc4, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_جوڑو_f32_f32(acc5, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_جوڑو_f32_f32(acc6, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_جوڑو_f32_f32(acc7, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_سمتیہ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_سمتیہ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_سمتیہ rA = hvx_vec_reduce_مجموع_f32x4(accA);
    HVX_سمتیہ rB = hvx_vec_reduce_مجموع_f32x4(accB);
    HVX_سمتیہ_Pred q16 = Q6_Q_vsetq2_R(16);
    واپس Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ساکن اندرونی HVX_سمتیہ gdn_جوڑو_scaled_dot8_f32(اعشاری * محدود dst0, اعشاری * محدود dst1,
        اعشاری * محدود dst2, اعشاری * محدود dst3, اعشاری * محدود dst4,
        اعشاری * محدود dst5, اعشاری * محدود dst6, اعشاری * محدود dst7,
        ثابت HVX_سمتیہ * محدود src, ثابت اعشاری * محدود scale,
        ثابت HVX_سمتیہ * محدود dot, uint32_t n) {
    HVX_سمتیہ acc0 = Q6_V_vzero();
    HVX_سمتیہ acc1 = Q6_V_vzero();
    HVX_سمتیہ acc2 = Q6_V_vzero();
    HVX_سمتیہ acc3 = Q6_V_vzero();
    HVX_سمتیہ acc4 = Q6_V_vzero();
    HVX_سمتیہ acc5 = Q6_V_vzero();
    HVX_سمتیہ acc6 = Q6_V_vzero();
    HVX_سمتیہ acc7 = Q6_V_vzero();
    ثابت HVX_سمتیہ scale0 = hvx_vec_splat_f32(scale[0]);
    ثابت HVX_سمتیہ scale1 = hvx_vec_splat_f32(scale[1]);
    ثابت HVX_سمتیہ scale2 = hvx_vec_splat_f32(scale[2]);
    ثابت HVX_سمتیہ scale3 = hvx_vec_splat_f32(scale[3]);
    ثابت HVX_سمتیہ scale4 = hvx_vec_splat_f32(scale[4]);
    ثابت HVX_سمتیہ scale5 = hvx_vec_splat_f32(scale[5]);
    ثابت HVX_سمتیہ scale6 = hvx_vec_splat_f32(scale[6]);
    ثابت HVX_سمتیہ scale7 = hvx_vec_splat_f32(scale[7]);

    ثابت uint32_t epv = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;
    برائے (uint32_t i = 0; i < nvec; ++i) {
        HVX_سمتیہ vs   = src[i];
        HVX_سمتیہ vdot = dot[i];

        HVX_سمتیہ out0 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_سمتیہ out1 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_سمتیہ out2 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_سمتیہ out3 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_سمتیہ out4 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_سمتیہ out5 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_سمتیہ out6 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_سمتیہ out7 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_جوڑو_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_جوڑو_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_جوڑو_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_جوڑو_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    اگر (nloe) {
        ثابت uint32_t off = nvec * epv;
        HVX_سمتیہ vs   = src[nvec];
        HVX_سمتیہ vdot = dot[nvec];
        HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(nloe * حجم(اعشاری));
        HVX_سمتیہ zero = Q6_V_vzero();

        HVX_سمتیہ out0 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_سمتیہ out1 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_سمتیہ out2 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_سمتیہ out3 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_سمتیہ out4 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_سمتیہ out5 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_سمتیہ out6 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_سمتیہ out7 = hvx_vec_جوڑو_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * حجم(اعشاری), out0);
        hvx_vec_store_u(dst1 + off, nloe * حجم(اعشاری), out1);
        hvx_vec_store_u(dst2 + off, nloe * حجم(اعشاری), out2);
        hvx_vec_store_u(dst3 + off, nloe * حجم(اعشاری), out3);
        hvx_vec_store_u(dst4 + off, nloe * حجم(اعشاری), out4);
        hvx_vec_store_u(dst5 + off, nloe * حجم(اعشاری), out5);
        hvx_vec_store_u(dst6 + off, nloe * حجم(اعشاری), out6);
        hvx_vec_store_u(dst7 + off, nloe * حجم(اعشاری), out7);

        acc0 = hvx_vec_جوڑو_f32_f32(acc0, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_جوڑو_f32_f32(acc1, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_جوڑو_f32_f32(acc2, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_جوڑو_f32_f32(acc3, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_جوڑو_f32_f32(acc4, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_جوڑو_f32_f32(acc5, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_جوڑو_f32_f32(acc6, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_جوڑو_f32_f32(acc7, Q6_V_vmux_QVV(نقاب, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_سمتیہ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_سمتیہ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_سمتیہ rA = hvx_vec_reduce_مجموع_f32x4(accA);
    HVX_سمتیہ rB = hvx_vec_reduce_مجموع_f32x4(accB);
    HVX_سمتیہ_Pred q16 = Q6_Q_vsetq2_R(16);
    واپس Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ساکن اندرونی خالی gdn_قدم_kda_f32(
    اعشاری * محدود s_work,
    اعشاری * محدود attn_برآمد,
    ثابت اعشاری * محدود q_t,
    ثابت اعشاری * محدود k_t,
    ثابت اعشاری * محدود v_t,
    ثابت اعشاری * محدود g_t,
    اعشاری beta_val,
    اعشاری scale,
    uint32_t S_v
) {
    ثابت uint32_t epv  = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = S_v / epv;
    ثابت uint32_t nloe = S_v % epv;

    HVX_سمتیہ vq[4];
    HVX_سمتیہ vk[4];
    HVX_سمتیہ vg[4];

    برائے (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    اگر (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
    }

    ثابت HVX_سمتیہ vbeta  = hvx_vec_splat_f32(beta_val);
    ثابت HVX_سمتیہ vscale = hvx_vec_splat_f32(scale);

    اعشاری delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    برائے (; j + 8 <= S_v; j += 8) {
        اعشاری * row0 = s_work + (uint64_t) (j + 0) * S_v;
        اعشاری * row1 = s_work + (uint64_t) (j + 1) * S_v;
        اعشاری * row2 = s_work + (uint64_t) (j + 2) * S_v;
        اعشاری * row3 = s_work + (uint64_t) (j + 3) * S_v;
        اعشاری * row4 = s_work + (uint64_t) (j + 4) * S_v;
        اعشاری * row5 = s_work + (uint64_t) (j + 5) * S_v;
        اعشاری * row6 = s_work + (uint64_t) (j + 6) * S_v;
        اعشاری * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_سمتیہ vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_سمتیہ vv_t   = hvx_vmemu(v_t + j);
        HVX_سمتیہ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_سمتیہ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * حجم(اعشاری), vdelta);

        HVX_سمتیہ vattn = gdn_جوڑو_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_سمتیہ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_برآمد + j, 8 * حجم(اعشاری), res_attn);
    }
    برائے (; j + 4 <= S_v; j += 4) {
        اعشاری * row0 = s_work + (uint64_t) (j + 0) * S_v;
        اعشاری * row1 = s_work + (uint64_t) (j + 1) * S_v;
        اعشاری * row2 = s_work + (uint64_t) (j + 2) * S_v;
        اعشاری * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_سمتیہ vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_سمتیہ vv_t   = hvx_vmemu(v_t + j);
        HVX_سمتیہ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_سمتیہ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * حجم(اعشاری), vdelta);

        HVX_سمتیہ vattn = gdn_جوڑو_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_سمتیہ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_برآمد + j, 4 * حجم(اعشاری), res_attn);
    }
    برائے (; j < S_v; ++j) {
        اعشاری * صف = s_work + (uint64_t) j * S_v;
        HVX_سمتیہ vsum = gdn_mul_dot_f32(صف, vg, vk, S_v);
        HVX_سمتیہ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_سمتیہ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_سمتیہ vres = gdn_جوڑو_scaled_dot_f32(صف, vk, vdj, vq, S_v);
        attn_برآمد[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ساکن اندرونی خالی gdn_قدم_scalar_f32(
    اعشاری * محدود s_work,
    اعشاری * محدود attn_برآمد,
    ثابت اعشاری * محدود q_t,
    ثابت اعشاری * محدود k_t,
    ثابت اعشاری * محدود v_t,
    ثابت اعشاری * محدود g_t,
    اعشاری beta_val,
    اعشاری scale,
    uint32_t S_v
) {
    ثابت uint32_t epv  = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = S_v / epv;
    ثابت uint32_t nloe = S_v % epv;

    HVX_سمتیہ vq[4];
    HVX_سمتیہ vk[4];

    برائے (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    اگر (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    ثابت HVX_سمتیہ vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    ثابت HVX_سمتیہ vbeta  = hvx_vec_splat_f32(beta_val);
    ثابت HVX_سمتیہ vscale = hvx_vec_splat_f32(scale);

    اعشاری delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    برائے (; j + 8 <= S_v; j += 8) {
        اعشاری * row0 = s_work + (uint64_t) (j + 0) * S_v;
        اعشاری * row1 = s_work + (uint64_t) (j + 1) * S_v;
        اعشاری * row2 = s_work + (uint64_t) (j + 2) * S_v;
        اعشاری * row3 = s_work + (uint64_t) (j + 3) * S_v;
        اعشاری * row4 = s_work + (uint64_t) (j + 4) * S_v;
        اعشاری * row5 = s_work + (uint64_t) (j + 5) * S_v;
        اعشاری * row6 = s_work + (uint64_t) (j + 6) * S_v;
        اعشاری * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_سمتیہ vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_سمتیہ vv_t   = hvx_vmemu(v_t + j);
        HVX_سمتیہ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_سمتیہ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * حجم(اعشاری), vdelta);

        HVX_سمتیہ vattn = gdn_جوڑو_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_سمتیہ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_برآمد + j, 8 * حجم(اعشاری), res_attn);
    }
    برائے (; j + 4 <= S_v; j += 4) {
        اعشاری * row0 = s_work + (uint64_t) (j + 0) * S_v;
        اعشاری * row1 = s_work + (uint64_t) (j + 1) * S_v;
        اعشاری * row2 = s_work + (uint64_t) (j + 2) * S_v;
        اعشاری * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_سمتیہ vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_سمتیہ vv_t   = hvx_vmemu(v_t + j);
        HVX_سمتیہ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_سمتیہ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * حجم(اعشاری), vdelta);

        HVX_سمتیہ vattn = gdn_جوڑو_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_سمتیہ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_برآمد + j, 4 * حجم(اعشاری), res_attn);
    }
    برائے (; j < S_v; ++j) {
        اعشاری * صف = s_work + (uint64_t) j * S_v;
        HVX_سمتیہ vsum = gdn_mul_scalar_dot_f32(صف, vgate, vk, S_v);
        HVX_سمتیہ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_سمتیہ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_سمتیہ vres = gdn_جوڑو_scaled_dot_f32(صف, vk, vdj, vq, S_v);
        attn_برآمد[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ساکن خالی gated_delta_net_f32_pp_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_gdn_context * gctx = (ساخت htp_gdn_context *) ڈیٹا;
    ساخت htp_ops_context * octx = gctx->octx;
    ثابت ساخت htp_gdn_kernel_params * kparams = gctx->kparams;

    ثابت ساخت htp_tensor * q     = octx->src[0];
    ثابت ساخت htp_tensor * k     = octx->src[1];
    ثابت ساخت htp_tensor * v     = octx->src[2];
    ثابت ساخت htp_tensor * g     = octx->src[3];
    ثابت ساخت htp_tensor * beta  = octx->src[4];
    ثابت ساخت htp_tensor * state = octx->src[5];
    ثابت ساخت htp_tensor * dst   = octx->dst;

    ثابت uint32_t S_v      = kparams->S_v;
    ثابت uint32_t H        = kparams->H;
    ثابت uint32_t n_علامات = kparams->n_علامات;
    ثابت uint32_t n_seqs   = kparams->n_seqs;
    ثابت uint32_t K        = kparams->K;
    ثابت uint32_t صف_انتہا  = gctx->صف_آغاز + gctx->nrows;

    اگر (ith >= gctx->nrows) {
        واپس;
    }

    ثابت ساخت htp_tensor * dst_cache = octx->dsts[1];
    ثابت اعشاری scale = kparams->scale;
    اعشاری * dst_بنیاد       = (اعشاری *) (uintptr_t) dst->ڈیٹا;
    اعشاری * state_برآمد_بنیاد = dst_cache ? (اعشاری *) (uintptr_t) dst_cache->ڈیٹا : (dst_بنیاد + S_v * H * n_علامات * n_seqs);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ثابت ساخت htp_gdn_vtcm_layout * layout = &gctx->layout;
    اعشاری * s_work[2];
    s_work[0] = (اعشاری *) (gctx->vtcm_بنیاد + layout->بائٹیں_فی_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / حجم(اعشاری);

    ثابت ساخت fastdiv_اقدار * fd_H   = &kparams->div_H;
    ثابت ساخت fastdiv_اقدار * fd_q1  = &kparams->div_q1;
    ثابت ساخت fastdiv_اقدار * fd_k1  = &kparams->div_k1;
    ثابت ساخت fastdiv_اقدار * fd_rq3 = &kparams->div_rq3;
    ثابت ساخت fastdiv_اقدار * fd_rk3 = &kparams->div_rk3;

    ثابت uint32_t state_seq_stride = kparams->state_seq_stride;
    ثابت uint64_t state_سائز_فی_snap = (uint64_t) kparams->state_سائز_فی_snap;
    ثابت dma_addr_t state_برآمد_dma_بنیاد = dst_cache ? dst_cache->ڈیٹا : (dst->ڈیٹا + S_v * H * n_علامات * n_seqs * حجم(اعشاری));

    uint32_t ir_prefetch = gctx->صف_آغاز + ith;
    صحیح spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    برائے (صحیح قدم = 0; قدم < 2 && ir_prefetch < صف_انتہا; قدم++) {
        ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_اندر  = state->ڈیٹا + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(اعشاری);
        dma_addr_t ps_برآمد = state_برآمد_dma_بنیاد + ((uint64_t) piv3 * H + piv1) * S_v * S_v * حجم(اعشاری);

        // Push dummy write-back
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(ps_برآمد, s_work[spad_idx]),
                       S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                       S_v * حجم(اعشاری), 0);

        // Push fetch
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_work[spad_idx], ps_اندر),
                       S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                       S_v * حجم(اعشاری), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    صحیح curr_spad_idx = 0;
    برائے (uint32_t ir = gctx->صف_آغاز + ith; ir < صف_انتہا; ir += nth) {
        dma_قطار_نکالو(dma_q);
        dma_قطار_نکالو(dma_q);

        اعشاری * s_work_curr = s_work[curr_spad_idx];

        ثابت uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ثابت uint32_t iv3 = fastdiv(ir, fd_H);

        ثابت uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ثابت uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ثابت uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ثابت uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_برآمد  = state_برآمد_dma_بنیاد + ((uint64_t) iv3 * H + iv1) * S_v * S_v * حجم(اعشاری);
        اعشاری * attn_ڈیٹا = dst_بنیاد + ((uint64_t) iv3 * n_علامات * H + iv1) * S_v;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        برائے (uint32_t t = 0; t < n_علامات; ++t) {
            ثابت اعشاری * q_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) q->ڈیٹا +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            ثابت اعشاری * k_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) k->ڈیٹا +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            ثابت اعشاری * v_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) v->ڈیٹا +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            ثابت اعشاری * g_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) g->ڈیٹا +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            ثابت اعشاری beta_val = *(ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) beta->ڈیٹا +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            اگر (kparams->kda) {
                gdn_قدم_kda_f32(s_work_curr, attn_ڈیٹا, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } ورنہ {
                gdn_قدم_scalar_f32(s_work_curr, attn_ڈیٹا, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            اگر (K > 1) {
                ثابت int64_t ہدف_خانہ = (int64_t) n_علامات - 1 - (int64_t) t;
                اگر (ہدف_خانہ > 0 && ہدف_خانہ < (int64_t) K) {
                    اعشاری * curr_state_o = state_برآمد_بنیاد + (uint64_t) ہدف_خانہ * state_سائز_فی_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (ثابت uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_ڈیٹا += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_برآمد, s_work_curr),
                       S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                       S_v * حجم(اعشاری), S_v);

        // Prefetch next block (if any)
        اگر (ir_prefetch < صف_انتہا) {
            ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_اندر = state->ڈیٹا + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(اعشاری);

            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_work[spad_idx], ps_اندر),
                           S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                           S_v * حجم(اعشاری), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_قطار_flush(dma_q);
}

ساکن خالی gated_delta_net_f32_tg_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_gdn_context * gctx = (ساخت htp_gdn_context *) ڈیٹا;
    ساخت htp_ops_context * octx = gctx->octx;
    ثابت ساخت htp_gdn_kernel_params * kparams = gctx->kparams;

    ثابت ساخت htp_tensor * q     = octx->src[0];
    ثابت ساخت htp_tensor * k     = octx->src[1];
    ثابت ساخت htp_tensor * v     = octx->src[2];
    ثابت ساخت htp_tensor * g     = octx->src[3];
    ثابت ساخت htp_tensor * beta  = octx->src[4];
    ثابت ساخت htp_tensor * state = octx->src[5];
    ثابت ساخت htp_tensor * dst   = octx->dst;

    ثابت uint32_t S_v      = kparams->S_v;
    ثابت uint32_t H        = kparams->H;
    ثابت uint32_t n_seqs   = kparams->n_seqs;
    ثابت uint32_t صف_انتہا  = gctx->صف_آغاز + gctx->nrows;

    اگر (ith >= gctx->nrows) {
        واپس;
    }

    ثابت ساخت htp_tensor * dst_cache = octx->dsts[1];
    ثابت اعشاری scale = kparams->scale;
    اعشاری * dst_بنیاد  = (اعشاری *) (uintptr_t) dst->ڈیٹا;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ثابت ساخت htp_gdn_vtcm_layout * layout = &gctx->layout;
    اعشاری * s_work[2];
    s_work[0] = (اعشاری *) (gctx->vtcm_بنیاد + layout->بائٹیں_فی_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / حجم(اعشاری);

    ثابت ساخت fastdiv_اقدار * fd_H   = &kparams->div_H;
    ثابت ساخت fastdiv_اقدار * fd_q1  = &kparams->div_q1;
    ثابت ساخت fastdiv_اقدار * fd_k1  = &kparams->div_k1;
    ثابت ساخت fastdiv_اقدار * fd_rq3 = &kparams->div_rq3;
    ثابت ساخت fastdiv_اقدار * fd_rk3 = &kparams->div_rk3;

    ثابت uint32_t state_seq_stride = kparams->state_seq_stride;
    ثابت dma_addr_t state_برآمد_dma_بنیاد = dst_cache ? dst_cache->ڈیٹا : (dst->ڈیٹا + S_v * H * n_seqs * حجم(اعشاری));

    uint32_t ir_prefetch = gctx->صف_آغاز + ith;
    صحیح spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    برائے (صحیح قدم = 0; قدم < 2 && ir_prefetch < صف_انتہا; قدم++) {
        ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_اندر  = state->ڈیٹا + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(اعشاری);
        dma_addr_t ps_برآمد = state_برآمد_dma_بنیاد + ((uint64_t) piv3 * H + piv1) * S_v * S_v * حجم(اعشاری);

        // Push dummy write-back
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(ps_برآمد, s_work[spad_idx]),
                       S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                       S_v * حجم(اعشاری), 0);

        // Push fetch
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_work[spad_idx], ps_اندر),
                       S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                       S_v * حجم(اعشاری), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    صحیح curr_spad_idx = 0;
    برائے (uint32_t ir = gctx->صف_آغاز + ith; ir < صف_انتہا; ir += nth) {
        dma_قطار_نکالو(dma_q);
        dma_قطار_نکالو(dma_q);

        اعشاری * s_work_curr = s_work[curr_spad_idx];

        ثابت uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ثابت uint32_t iv3 = fastdiv(ir, fd_H);

        ثابت uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ثابت uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ثابت uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ثابت uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_برآمد  = state_برآمد_dma_بنیاد + ((uint64_t) iv3 * H + iv1) * S_v * S_v * حجم(اعشاری);
        اعشاری * attn_ڈیٹا = dst_بنیاد + ((uint64_t) iv3 * H + iv1) * S_v;

        ثابت اعشاری * q_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) q->ڈیٹا +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        ثابت اعشاری * k_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) k->ڈیٹا +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        ثابت اعشاری * v_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) v->ڈیٹا +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        ثابت اعشاری * g_t = (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) g->ڈیٹا +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        ثابت اعشاری beta_val = *(ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) beta->ڈیٹا +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        اگر (kparams->kda) {
            gdn_قدم_kda_f32(s_work_curr, attn_ڈیٹا, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } ورنہ {
            gdn_قدم_scalar_f32(s_work_curr, attn_ڈیٹا, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_برآمد, s_work_curr),
                       S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                       S_v * حجم(اعشاری), S_v);

        // Prefetch next block (if any)
        اگر (ir_prefetch < صف_انتہا) {
            ثابت uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ثابت uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_اندر = state->ڈیٹا + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * حجم(اعشاری);

            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_work[spad_idx], ps_اندر),
                           S_v * حجم(اعشاری), S_v * حجم(اعشاری),
                           S_v * حجم(اعشاری), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_قطار_flush(dma_q);
}

ساخت htp_gdn_hmx_gemm_task {
    ثابت __fp16 * صف_tiles;
    ثابت __fp16 * ستون_tiles;
    __fp16 *       برآمد_tiles;
    uint32_t       n_صف_tiles;
    uint32_t       n_ستون_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

ساکن خالی htp_gdn_hmx_gemm_کارکن(خالی * ڈیٹا) {
    ساخت htp_gdn_hmx_gemm_task * task = (ساخت htp_gdn_hmx_gemm_task *) ڈیٹا;
    asm متلون(HMX_سیٹ_BIAS("%0") :: "r"((غیرعلامتی صحیح)task->hmx_scales));

    ثابت size_t dot_stride = task->dot_stride;
    برائے (uint32_t r = 0; r < task->n_صف_tiles; ++r) {
        ثابت __fp16 * r_tiles = task->صف_tiles + r * dot_stride;
        ثابت __fp16 * c_tiles = task->ستون_tiles;
        __fp16 *       o_tile  = task->برآمد_tiles + r * task->n_ستون_tiles * HMX_FP16_TILE_N_ELMS;

        برائے (uint32_t c = 0; c < task->n_ستون_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

ساکن اندرونی خالی htp_gdn_ڈالو_hmx_gemm_task(
    hmx_قطار_t q,
    ساخت htp_gdn_hmx_gemm_task * task,
    ثابت __fp16 * صف_tiles,
    ثابت __fp16 * ستون_tiles,
    __fp16 * برآمد_tiles,
    uint32_t n_صف_tiles,
    uint32_t n_ستون_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    task->صف_tiles   = صف_tiles;
    task->ستون_tiles   = ستون_tiles;
    task->برآمد_tiles   = برآمد_tiles;
    task->n_صف_tiles = n_صف_tiles;
    task->n_ستون_tiles = n_ستون_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    task->hmx_scales  = scales;

    hmx_قطار_ڈالو(q, hmx_قطار_make_desc(htp_gdn_hmx_gemm_کارکن, task));
}

ساکن اندرونی خالی gdn_unpack_64x64_tiles_مقصد_vectors(
    HVX_سمتیہ * محدود صفیں,
    ثابت __fp16 * محدود tiles
) {
    ثابت HVX_سمتیہ * t00 = (ثابت HVX_سمتیہ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    ثابت HVX_سمتیہ * t01 = (ثابت HVX_سمتیہ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    ثابت HVX_سمتیہ * t10 = (ثابت HVX_سمتیہ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    ثابت HVX_سمتیہ * t11 = (ثابت HVX_سمتیہ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    برائے (uint32_t r = 0; r < 16; ++r) {
        HVX_سمتیہ_Pair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        صفیں[2 * r + 0] = Q6_V_lo_W(vp0);
        صفیں[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_سمتیہ_Pair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        صفیں[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        صفیں[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

ساکن اندرونی خالی gdn_pack_64x64_vectors_مقصد_tiles(
    __fp16 * محدود tiles,
    ثابت HVX_سمتیہ * محدود صفیں
) {
    HVX_سمتیہ * t00 = (HVX_سمتیہ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_سمتیہ * t01 = (HVX_سمتیہ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_سمتیہ * t10 = (HVX_سمتیہ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_سمتیہ * t11 = (HVX_سمتیہ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    برائے (uint32_t r = 0; r < 16; ++r) {
        HVX_سمتیہ_Pair vp0 = Q6_W_vshuff_VVR(صفیں[2 * r + 1], صفیں[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_سمتیہ_Pair vp1 = Q6_W_vshuff_VVR(صفیں[32 + 2 * r + 1], صفیں[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

ساکن اندرونی خالی gdn_unpack_64x_S_tiles_مقصد_f32(
    اعشاری * محدود dst_f32,
    ثابت __fp16 * محدود tiles,
    uint32_t S_v
) {
    ثابت uint32_t n_ستون_tiles = S_v / 32;
    برائے (uint32_t r0 = 0; r0 < 2; ++r0) {
        برائے (uint32_t d = 0; d < S_v / 64; ++d) {
            ثابت HVX_سمتیہ * t0 = (ثابت HVX_سمتیہ *) (tiles + (r0 * n_ستون_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ثابت HVX_سمتیہ * t1 = (ثابت HVX_سمتیہ *) (tiles + (r0 * n_ستون_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            برائے (uint32_t r = 0; r < 16; ++r) {
                HVX_سمتیہ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_سمتیہ_Pair p0 = hvx_vec_f16_مقصد_f32(Q6_V_lo_W(vp01));
                HVX_سمتیہ_Pair p1 = hvx_vec_f16_مقصد_f32(Q6_V_hi_W(vp01));

                اعشاری * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                اعشاری * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ساکن اندرونی خالی gdn_unpack_64x_S_tiles_مقصد_f16(
    __fp16 * محدود dst_f16,
    ثابت __fp16 * محدود tiles,
    uint32_t S_v
) {
    ثابت uint32_t n_ستون_tiles = S_v / 32;
    برائے (uint32_t r0 = 0; r0 < 2; ++r0) {
        برائے (uint32_t d = 0; d < S_v / 64; ++d) {
            ثابت HVX_سمتیہ * t0 = (ثابت HVX_سمتیہ *) (tiles + (r0 * n_ستون_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ثابت HVX_سمتیہ * t1 = (ثابت HVX_سمتیہ *) (tiles + (r0 * n_ستون_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            برائے (uint32_t r = 0; r < 16; ++r) {
                HVX_سمتیہ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ساکن اندرونی خالی gdn_unpack_Sx_S_tiles_مقصد_f32(
    اعشاری * محدود dst_f32,
    ثابت __fp16 * محدود tiles,
    uint32_t S_v
) {
    ثابت uint32_t n_tiles = S_v / 32;
    برائے (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        برائے (uint32_t d = 0; d < S_v / 64; ++d) {
            ثابت HVX_سمتیہ * t0 = (ثابت HVX_سمتیہ *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ثابت HVX_سمتیہ * t1 = (ثابت HVX_سمتیہ *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            برائے (uint32_t r = 0; r < 16; ++r) {
                HVX_سمتیہ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_سمتیہ_Pair p0 = hvx_vec_f16_مقصد_f32(Q6_V_lo_W(vp01));
                HVX_سمتیہ_Pair p1 = hvx_vec_f16_مقصد_f32(Q6_V_hi_W(vp01));

                اعشاری * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                اعشاری * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ساکن اندرونی خالی gdn_f32_مقصد_hmx_صف_tiles_and_f16(
    __fp16 * محدود dst_tiles,
    __fp16 * محدود dst_مفرد_tiles,
    __fp16 * محدود dst_f16,
    ثابت اعشاری * محدود src,
    ثابت __fp16 * محدود scale_فی_صف,
    uint32_t n_صفیں,
    uint32_t n_ستون_تعداد
) {
    ثابت uint32_t n_ستون_tiles = n_ستون_تعداد / 32;
    ثابت uint32_t * scale_جوڑے = (ثابت uint32_t *) scale_فی_صف;

    برائے (uint32_t r = 0; r < n_صفیں; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        ثابت اعشاری * p0 = src + (r + 0) * n_ستون_تعداد;
        ثابت اعشاری * p1 = src + (r + 1) * n_ستون_تعداد;

        HVX_سمتیہ v_scale;
        اگر (dst_مفرد_tiles) {
            uint32_t scale_pair = scale_جوڑے ? scale_جوڑے[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        برائے (uint32_t c = 0; c < n_ستون_tiles; c += 2) {
            HVX_سمتیہ v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_سمتیہ v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_سمتیہ v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_سمتیہ v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_سمتیہ vh0 = hvx_vec_f32_مقصد_f16_shuff(v0_0, v1_0);
            HVX_سمتیہ vh1 = hvx_vec_f32_مقصد_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_ستون_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_ستون_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_سمتیہ *) tile0)[r1] = vh0;
            ((HVX_سمتیہ *) tile1)[r1] = vh1;

            اگر (dst_مفرد_tiles) {
                HVX_سمتیہ vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_سمتیہ vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_مفرد_tiles + (r0 * n_ستون_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_مفرد_tiles + (r0 * n_ستون_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_سمتیہ *) tile0_s)[r1] = vh0_s;
                ((HVX_سمتیہ *) tile1_s)[r1] = vh1_s;
            }

            اگر (dst_f16) {
                HVX_سمتیہ_Pair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_ستون_تعداد + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_ستون_تعداد + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ساکن اندرونی خالی hvx_transpose_32x32_الفاظ(HVX_سمتیہ * محدود m, HVX_سمتیہ * محدود tmp) {
    برائے (صحیح i = 0; i < 16; ++i) {
        HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    برائے (صحیح b = 0; b < 32; b += 4) {
        HVX_سمتیہ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_سمتیہ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    برائے (صحیح b = 0; b < 32; b += 8) {
        برائے (صحیح i = 0; i < 4; ++i) {
            HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    برائے (صحیح b = 0; b < 32; b += 16) {
        برائے (صحیح i = 0; i < 8; ++i) {
            HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    برائے (صحیح i = 0; i < 16; ++i) {
        HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    برائے (صحیح i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

ساکن اندرونی خالی gdn_pack_d_t_صف_tiles(
    __fp16 * محدود dst_tiles,
    ثابت __fp16 * محدود src_d,
    uint32_t S_v,
    HVX_سمتیہ * محدود m,
    HVX_سمتیہ * محدود tmp
) {
    برائے (uint32_t ستون_half = 0; ستون_half < S_v / 64; ++ستون_half) {
        uint32_t r0_بنیاد = ستون_half * 2;
        برائے (uint32_t c0 = 0; c0 < 2; ++c0) {
            برائے (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + ستون_half * 64);
            }

            hvx_transpose_32x32_الفاظ(m, tmp);

            uint32_t tile0_idx = (r0_بنیاد + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_بنیاد + 1) * 2 + c0;
            HVX_سمتیہ * t0 = (HVX_سمتیہ *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_سمتیہ * t1 = (HVX_سمتیہ *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            برائے (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

ساکن __attribute__((noinline)) خالی gdn_build_inv_l_بلاکیں(
    __fp16 * محدود inv_صف_tiles,
    ثابت HVX_سمتیہ * محدود صفیں_kk,
    ثابت __fp16 * محدود decay_m,
    ثابت اعشاری * محدود beta,
    __fp16 * محدود l10_tile,
    __fp16 * محدود neg_a11_tile
) {
    ثابت HVX_سمتیہ v_one_f16 = hvx_vec_splat_f16(1.0f);
    ثابت HVX_سمتیہ_Pred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_سمتیہ * محدود p_l00 = (HVX_سمتیہ *) l00;
    HVX_سمتیہ * محدود p_l11 = (HVX_سمتیہ *) l11;
    HVX_سمتیہ * محدود p_l10_tile = (HVX_سمتیہ *) l10_tile;

    HVX_سمتیہ * محدود tile00 = (HVX_سمتیہ *) (inv_صف_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_سمتیہ * محدود tile01 = (HVX_سمتیہ *) (inv_صف_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_سمتیہ * محدود tile11 = (HVX_سمتیہ *) (inv_صف_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_سمتیہ * محدود p_neg_a11 = (HVX_سمتیہ *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_مقصد_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    برائے (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    برائے (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_سمتیہ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_سمتیہ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_سمتیہ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_سمتیہ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_سمتیہ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفیں_kk[t0], v_d0), v_b0);
        HVX_سمتیہ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفیں_kk[t1], v_d1), v_b1);

        p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
    }

    برائے (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 32 + 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_سمتیہ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_سمتیہ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_سمتیہ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_سمتیہ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_سمتیہ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفیں_kk[t0], v_d0), v_b0);
        HVX_سمتیہ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(صفیں_kk[t1], v_d1), v_b1);

        HVX_سمتیہ_Pair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
        p_l10_tile[r] = Q6_V_lo_W(vp_l10);
        p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
    }

    HVX_سمتیہ a_صفیں[32];
    برائے (uint32_t t = 0; t < 32; ++t) {
        HVX_سمتیہ v_inv = Q6_V_vzero();
        برائے (uint32_t k = 0; k < t; ++k) {
            HVX_سمتیہ v_lk = Q6_Vh_vsplat_R(l00[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_صفیں[k]));
        }
        HVX_سمتیہ_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_صفیں[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    برائے (uint32_t r = 0; r < 16; ++r) {
        HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(a_صفیں[2 * r + 1], a_صفیں[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    برائے (uint32_t t = 0; t < 32; ++t) {
        HVX_سمتیہ v_inv = Q6_V_vzero();
        برائے (uint32_t k = 0; k < t; ++k) {
            HVX_سمتیہ v_lk = Q6_Vh_vsplat_R(l11[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_صفیں[k]));
        }
        HVX_سمتیہ_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_صفیں[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    برائے (uint32_t r = 0; r < 16; ++r) {
        HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(a_صفیں[2 * r + 1], a_صفیں[2 * r + 0], -2);
        tile11[r] = Q6_V_lo_W(vp);

        HVX_سمتیہ n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_صفیں[2 * r + 0]);
        HVX_سمتیہ n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_صفیں[2 * r + 1]);
        HVX_سمتیہ_Pair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
        p_neg_a11[r] = Q6_V_lo_W(vp_neg);
    }
}


ساکن اندرونی خالی gdn_dma_ڈالو_chunk_inputs(
    dma_قطار * dma_q,
    اعشاری * vtcm_q,
    اعشاری * vtcm_k,
    اعشاری * vtcm_v,
    ثابت ساخت htp_tensor * q,
    ثابت ساخت htp_tensor * k,
    ثابت ساخت htp_tensor * v,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_سائز,
    uint32_t S_v
) {
    ثابت dma_addr_t q_dma = q->ڈیٹا + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
    ثابت dma_addr_t k_dma = k->ڈیٹا + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
    ثابت dma_addr_t v_dma = v->ڈیٹا + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_q, q_dma), S_v * حجم(اعشاری), q->nb[2], S_v * حجم(اعشاری), chunk_سائز);
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_k, k_dma), S_v * حجم(اعشاری), k->nb[2], S_v * حجم(اعشاری), chunk_سائز);
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_v, v_dma), S_v * حجم(اعشاری), v->nb[2], S_v * حجم(اعشاری), chunk_سائز);
}

ساکن اندرونی خالی gdn_dma_ڈالو_chunk_gb(
    dma_قطار * dma_q,
    اعشاری * vtcm_g_raw,
    اعشاری * vtcm_b_raw,
    ثابت ساخت htp_tensor * g,
    ثابت ساخت htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_سائز,
    uint32_t n_batch
) {
    ثابت dma_addr_t g_dma    = g->ڈیٹا + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    ثابت dma_addr_t beta_dma = beta->ڈیٹا + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    ثابت uint32_t صف_بائٹیں  = n_batch * حجم(اعشاری);

    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_g_raw, g_dma), صف_بائٹیں, g->nb[2], صف_بائٹیں, chunk_سائز);
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_b_raw, beta_dma), صف_بائٹیں, beta->nb[2], صف_بائٹیں, chunk_سائز);
}

ساکن اندرونی خالی gdn_pack_s_ستون_tiles(
    __fp16 * محدود vtcm_s_ستون_tiles,
    __fp16 * محدود vtcm_s_f16,
    ثابت اعشاری * محدود vtcm_s_state,
    uint32_t S_v
) {
    برائے (uint32_t j = 0; j < S_v; ++j) {
        برائے (uint32_t i = 0; i < S_v; i += 64) {
            HVX_سمتیہ v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_سمتیہ v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_مقصد_f16(v0, v1);
        }
    }
    hmx_interleave_صفیں_مقصد_tiles(vtcm_s_ستون_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);
}

ساخت htp_gdn_سر_ptrs {
    اعشاری *  s_state;
    __fp16 * s_f16;
    __fp16 * s_ستون_tiles;
    اعشاری *  s_update_f32;
    __fp16 * s_update_tiles;

    اعشاری * q_f32[2];
    اعشاری * k_f32[2];
    اعشاری * v_f32[2];
    اعشاری * g_f32[2];
    اعشاری * b_f32[2];
    اعشاری * o_f32[2];

    اعشاری * v_inter_f32;
    اعشاری * o_inter_f32;
    اعشاری * o_intra_f32;

    __fp16 * k_f16;
    __fp16 * v_مفرد_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_صف_tiles;
    __fp16 * q_مفرد_صف_tiles;
    __fp16 * k_صف_tiles;
    __fp16 * k_ستون_tiles;
    __fp16 * k_مفرد_صف_tiles;
    __fp16 * k_ستون_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_صف_tiles;
    __fp16 * a_صف_tiles;
    __fp16 * v_مفرد_ستون_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_ستون_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_صف_tiles;

    __fp16 * gamma;
    اعشاری *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_سمتیہ * صفیں_kk;
    HVX_سمتیہ * صفیں_qk;
    HVX_سمتیہ * صفیں_inv;
    HVX_سمتیہ * صفیں_a;

    HVX_سمتیہ * vtcm_m;
    HVX_سمتیہ * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_اندر_dma;
    dma_addr_t state_برآمد_dma;
};

ساکن اندرونی خالی gdn_init_سر_ptrs(
    ساخت htp_gdn_سر_ptrs * سر,
    ثابت ساخت htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_بنیاد,
    uint32_t h,
    uint32_t بنیاد_iv1,
    uint32_t iv3,
    ثابت ساخت htp_tensor * q,
    ثابت ساخت htp_tensor * k,
    ثابت ساخت htp_tensor * v,
    ثابت ساخت htp_tensor * state,
    ثابت ساخت htp_tensor * dst,
    ثابت ساخت htp_tensor * dst_cache,
    ثابت ساخت htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_علامات,
    uint32_t chunk_سائز
) {
    ثابت size_t dma_scalar_sz = hex_باری_اوپر(chunk_سائز * حجم(اعشاری), 128);
    ثابت size_t decay_sz      = 64 * 64 * حجم(__fp16);
    ثابت size_t صف_vecs_sz   = 64 * 128;

    سر->s_state        = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_s_state + h * L->state_f32_بائٹیں);
    سر->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_s_f16 + h * L->state_f16_بائٹیں);
    سر->s_ستون_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_s_ستون_tiles + h * L->state_tiles_بائٹیں);
    سر->s_update_f32   = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_s_update_f32 + h * L->state_f32_بائٹیں);
    سر->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_s_update_tiles + h * L->state_tiles_بائٹیں);

    سر->q_f32[0] = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_q_f32[0] + h * L->dma_chunk_بائٹیں);
    سر->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_q_f32[1] + h * L->dma_chunk_بائٹیں) : سر->q_f32[0];
    سر->k_f32[0] = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_k_f32[0] + h * L->dma_chunk_بائٹیں);
    سر->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_k_f32[1] + h * L->dma_chunk_بائٹیں) : سر->k_f32[0];
    سر->v_f32[0] = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_v_f32[0] + h * L->dma_chunk_بائٹیں);
    سر->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_v_f32[1] + h * L->dma_chunk_بائٹیں) : سر->v_f32[0];
    سر->g_f32[0] = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_g_f32[0] + h * dma_scalar_sz);
    سر->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_g_f32[1] + h * dma_scalar_sz) : سر->g_f32[0];
    سر->b_f32[0] = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_b_f32[0] + h * dma_scalar_sz);
    سر->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_b_f32[1] + h * dma_scalar_sz) : سر->b_f32[0];
    سر->o_f32[0] = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_o_f32[0] + h * L->dma_chunk_بائٹیں);
    سر->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_o_f32[1] + h * L->dma_chunk_بائٹیں) : سر->o_f32[0];

    سر->v_inter_f32 = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_v_inter_f32 + h * L->dma_chunk_بائٹیں);
    سر->o_inter_f32 = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_o_inter_f32 + h * L->dma_chunk_بائٹیں);
    سر->o_intra_f32 = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_o_intra_f32 + h * L->dma_chunk_بائٹیں);

    سر->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_k_f16 + h * L->act_f16_بائٹیں);
    سر->v_مفرد_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_v_مفرد_f16 + h * L->act_f16_بائٹیں);
    سر->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_delta_f16 + h * L->act_f16_بائٹیں);
    سر->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_d_f16 + h * L->act_f16_بائٹیں);

    سر->q_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_q_صف_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->q_مفرد_صف_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_q_مفرد_صف_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->k_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_k_صف_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->k_ستون_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_k_ستون_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->k_مفرد_صف_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_k_مفرد_صف_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->k_ستون_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_k_ستون_tiles_64x128 + h * L->tile_64x_Sv_بائٹیں);
    سر->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_kk_tiles + h * L->tile_64x64_بائٹیں);
    سر->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_qk_tiles + h * L->tile_64x64_بائٹیں);
    سر->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_v_inter_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_o_inter_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->inv_صف_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_inv_صف_tiles + h * L->tile_64x64_بائٹیں);
    سر->a_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_a_صف_tiles + h * L->tile_64x64_بائٹیں);
    سر->v_مفرد_ستون_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_v_مفرد_ستون_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_delta_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->delta_ستون_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_delta_ستون_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_o_intra_tiles + h * L->tile_64x_Sv_بائٹیں);
    سر->d_صف_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_d_صف_tiles + h * L->tile_64x_Sv_بائٹیں);

    سر->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_gamma + h * dma_scalar_sz);
    سر->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_gamma + h * dma_scalar_sz + 128);
    سر->lambda_init     = VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L->off_lambda_init + h * dma_scalar_sz);
    سر->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_decay_m + h * decay_sz);
    سر->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_بنیاد, L->off_decay_a + h * decay_sz);

    سر->صفیں_kk  = VTCM_LAYOUT_PTR(HVX_سمتیہ, vtcm_بنیاد, L->off_صفیں_kk + h * صف_vecs_sz);
    سر->صفیں_qk  = VTCM_LAYOUT_PTR(HVX_سمتیہ, vtcm_بنیاد, L->off_صفیں_qk + h * صف_vecs_sz);
    سر->صفیں_inv = VTCM_LAYOUT_PTR(HVX_سمتیہ, vtcm_بنیاد, L->off_صفیں_inv + h * صف_vecs_sz);
    سر->صفیں_a   = VTCM_LAYOUT_PTR(HVX_سمتیہ, vtcm_بنیاد, L->off_صفیں_a + h * صف_vecs_sz);

    سر->vtcm_m   = VTCM_LAYOUT_PTR(HVX_سمتیہ, vtcm_بنیاد, L->off_thread_scratch + h * (64 * 128));
    سر->vtcm_tmp = سر->vtcm_m + 32;

    سر->iv1 = بنیاد_iv1 + h;
    سر->iv3 = iv3;
    سر->iq1 = fastmodulo(سر->iv1, q->ne[1], &kparams->div_q1);
    سر->ik1 = fastmodulo(سر->iv1, k->ne[1], &kparams->div_k1);
    سر->iq3 = fastdiv(سر->iv3, &kparams->div_rq3);
    سر->ik3 = fastdiv(سر->iv3, &kparams->div_rk3);

    سر->state_اندر_dma = state->ڈیٹا +
        ((uint64_t) سر->iv3 * kparams->state_seq_stride + (uint64_t) سر->iv1 * S_v * S_v) * حجم(اعشاری);

    سر->state_برآمد_dma = dst_cache ?
        (dst_cache->ڈیٹا + ((uint64_t) سر->iv3 * H + سر->iv1) * S_v * S_v * حجم(اعشاری)) :
        (dst->ڈیٹا + ((uint64_t) S_v * H * n_علامات * kparams->n_seqs + (uint64_t) (سر->iv3 * H + سر->iv1) * S_v * S_v) * حجم(اعشاری));
}

ساخت htp_gdn_batch_context {
    ساخت htp_gdn_سر_ptrs * heads;
    ثابت اعشاری *              vtcm_g_raw;
    ثابت اعشاری *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    اعشاری                      scale;
    ساخت htp_ops_context *   octx;
    ثابت ساخت htp_gdn_kernel_params * kparams;
};

ساکن خالی gdn_hvx_init_state_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    gdn_pack_s_ستون_tiles(سر->s_ستون_tiles, سر->s_f16, سر->s_state, bctx->S_v);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

ساکن اندرونی __attribute__((غیر_مستعمل_2)) HVX_سمتیہ hvx_clamp_neg20_0(HVX_سمتیہ v, HVX_سمتیہ v_zero, HVX_سمتیہ v_neg20) {
    HVX_سمتیہ_Pred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_سمتیہ_Pred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    واپس Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

ساکن اندرونی HVX_سمتیہ hvx_سابقہ_جائزہ_f32(HVX_سمتیہ v, HVX_سمتیہ حاصل_اندر) {
    ثابت HVX_سمتیہ zero = Q6_V_vzero();

    v = hvx_vec_جوڑو_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_جوڑو_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_جوڑو_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_جوڑو_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_جوڑو_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_جوڑو_f32_f32(v, حاصل_اندر);

    واپس v;
}

ساکن اندرونی HVX_سمتیہ hvx_splat_آخری_f32(HVX_سمتیہ v) {
    واپس hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساکن خالی gdn_hvx_phase1a_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;
    ثابت uint32_t n_batch = bctx->n_batch;

    اگر (n_batch == 1) {
        hvx_vmem(سر->g_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(سر->g_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(سر->b_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(سر->b_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } ورنہ {
        int32_t offsets[32] __attribute__((aligned(128)));
        برائے (صحیح k = 0; k < 32; ++k) {
            offsets[k] = k * n_batch * حجم(اعشاری);
        }
        HVX_سمتیہ vv = *(ثابت HVX_سمتیہ *) offsets;
        ثابت size_t rt_g = (size_t) ((ثابت uint8_t *) bctx->vtcm_g_raw + i * حجم(اعشاری));
        ثابت size_t rt_b = (size_t) ((ثابت uint8_t *) bctx->vtcm_b_raw + i * حجم(اعشاری));
        ثابت size_t mu   = 64 * n_batch * حجم(اعشاری);

        Q6_vgather_ARMVw((HVX_سمتیہ *) (سر->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_سمتیہ *) (سر->g_f32[curr_buf] + 32), rt_g + 32 * n_batch * حجم(اعشاری), mu, vv);
        Q6_vgather_ARMVw((HVX_سمتیہ *) (سر->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_سمتیہ *) (سر->b_f32[curr_buf] + 32), rt_b + 32 * n_batch * حجم(اعشاری), mu, vv);
    }

    ثابت uint32_t t_chunk = bctx->c * 64;
    ثابت uint32_t valid_علامات = hex_smin(64, bctx->kparams->n_علامات - t_chunk);
    اگر (valid_علامات < 64) {
        برائے (uint32_t t = valid_علامات; t < 64; ++t) {
            سر->g_f32[curr_buf][t] = 0.0f;
            سر->b_f32[curr_buf][t] = 0.0f;
        }
        ثابت HVX_سمتیہ vzero = Q6_V_vzero();
        برائے (uint32_t t = valid_علامات; t < 64; ++t) {
            برائے (uint32_t j = 0; j < S_v; j += 32) {
                hvx_vmem(سر->q_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(سر->k_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(سر->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    ثابت HVX_سمتیہ v_g0 = hvx_vmem(سر->g_f32[curr_buf] + 0);
    ثابت HVX_سمتیہ v_g1 = hvx_vmem(سر->g_f32[curr_buf] + 32);

    HVX_سمتیہ v_gamma0 = hvx_سابقہ_جائزہ_f32(v_g0, Q6_V_vzero());
    HVX_سمتیہ v_حاصل  = hvx_splat_آخری_f32(v_gamma0);
    HVX_سمتیہ v_gamma1 = hvx_سابقہ_جائزہ_f32(v_g1, v_حاصل);

    ثابت HVX_سمتیہ v_zero  = Q6_V_vzero();
    ثابت HVX_سمتیہ v_neg20 = hvx_vec_splat_f32(-20.0f);

    hvx_vmem(سر->gamma) = hvx_vec_f32_مقصد_f16(v_gamma0, v_gamma1);

    HVX_سمتیہ v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_سمتیہ v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    hvx_vmem(سر->lambda_init + 0)  = v_l0;
    hvx_vmem(سر->lambda_init + 32) = v_l1;
    hvx_vmem(سر->lambda_init_f16)  = hvx_vec_f32_مقصد_f16(v_l0, v_l1);

    gdn_f32_مقصد_hmx_صف_tiles_and_f16(سر->k_صف_tiles, سر->k_مفرد_صف_tiles, سر->k_f16,
                                     سر->k_f32[curr_buf], سر->lambda_init_f16, 64, S_v);
    hmx_interleave_صفیں_مقصد_tiles(سر->k_ستون_tiles, سر->k_f16, 64, S_v, S_v, 0, 64);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ساکن خالی gdn_hvx_phase1b_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;

    gdn_f32_مقصد_hmx_صف_tiles_and_f16(سر->q_صف_tiles, سر->q_مفرد_صف_tiles, تہی,
                                     سر->q_f32[curr_buf], سر->lambda_init_f16, 64, S_v);

    hmx_interleave_ستون_تعداد_مقصد_tiles(سر->k_ستون_tiles_64x128, سر->k_f16, 64, S_v, S_v, 2, 0, 64);

    ثابت uint16_t * gamma_u16 = (ثابت uint16_t *) سر->gamma;
    ثابت HVX_سمتیہ v_gamma   = hvx_vmem(سر->gamma);

    ثابت HVX_سمتیہ v_zero_f16  = Q6_V_vzero();
    ثابت HVX_سمتیہ v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    ثابت HVX_سمتیہ v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    ثابت HVX_سمتیہ v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(سر->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(سر->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    برائے (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_سمتیہ v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_سمتیہ v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_سمتیہ diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_سمتیہ diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_سمتیہ_Pred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_سمتیہ_Pred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_سمتیہ diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_سمتیہ diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_سمتیہ v_exp0      = hvx_vec_exp2_f16(diff_log2e0);
        HVX_سمتیہ v_exp1      = hvx_vec_exp2_f16(diff_log2e1);

        HVX_سمتیہ_Pred نقاب_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_سمتیہ_Pred نقاب_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_سمتیہ v_m0         = Q6_V_vand_QV(نقاب_lt0, v_exp0);
        HVX_سمتیہ v_m1         = Q6_V_vand_QV(نقاب_lt1, v_exp1);

        HVX_سمتیہ_Pred نقاب_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_سمتیہ_Pred نقاب_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_سمتیہ_Pred نقاب_diag0 = Q6_Q_and_QQn(نقاب_le0, نقاب_lt0);
        HVX_سمتیہ_Pred نقاب_diag1 = Q6_Q_and_QQn(نقاب_le1, نقاب_lt1);

        HVX_سمتیہ v_a0         = Q6_V_vmux_QVV(نقاب_diag0, v_one_f16, v_m0);
        HVX_سمتیہ v_a1         = Q6_V_vmux_QVV(نقاب_diag1, v_one_f16, v_m1);

        hvx_vmem(سر->decay_m + t0 * 64) = v_m0;
        hvx_vmem(سر->decay_a + t0 * 64) = v_a0;
        hvx_vmem(سر->decay_m + t1 * 64) = v_m1;
        hvx_vmem(سر->decay_a + t1 * 64) = v_a1;
    }

    {
        HVX_سمتیہ v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_سمتیہ diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_سمتیہ_Pred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_سمتیہ diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_سمتیہ v_exp         = hvx_vec_exp2_f16(diff_log2e);

        HVX_سمتیہ_Pred نقاب_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_سمتیہ v_m           = Q6_V_vand_QV(نقاب_lt_t, v_exp);

        HVX_سمتیہ_Pred نقاب_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_سمتیہ_Pred نقاب_diag = Q6_Q_and_QQn(نقاب_le_t, نقاب_lt_t);
        HVX_سمتیہ v_a           = Q6_V_vmux_QVV(نقاب_diag, v_one_f16, v_m);

        hvx_vmem(سر->decay_m + 63 * 64) = v_m;
        hvx_vmem(سر->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ساکن خالی gdn_hvx_phase2_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;

    gdn_unpack_64x64_tiles_مقصد_vectors(سر->صفیں_kk, سر->kk_tiles);

    gdn_build_inv_l_بلاکیں(
        سر->inv_صف_tiles,
        سر->صفیں_kk,
        سر->decay_m,
        سر->b_f32[curr_buf],
        (__fp16 *) سر->vtcm_m,
        (__fp16 *) سر->vtcm_m + HMX_FP16_TILE_N_ELMS
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

ساکن خالی gdn_hvx_phase3_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_مقصد_f32(سر->v_inter_f32, سر->v_inter_tiles, S_v);

    HVX_سمتیہ_Alias local_b[2];
    local_b[0].v = hvx_vmem(سر->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(سر->b_f32[curr_buf] + 32);

    برائے (uint32_t t = 0; t < 64; ++t) {
        HVX_سمتیہ vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
        برائے (uint32_t j = 0; j < S_v; j += 64) {
            HVX_سمتیہ vv0 = hvx_vmem(سر->v_f32[curr_buf] + t * S_v + j + 0);
            HVX_سمتیہ vv1 = (j + 32 < S_v) ? hvx_vmem(سر->v_f32[curr_buf] + t * S_v + j + 32) : Q6_V_vzero();
            HVX_سمتیہ vi0 = hvx_vmem(سر->v_inter_f32 + t * S_v + j + 0);
            HVX_سمتیہ vi1 = (j + 32 < S_v) ? hvx_vmem(سر->v_inter_f32 + t * S_v + j + 32) : Q6_V_vzero();

            HVX_سمتیہ vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
            HVX_سمتیہ vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

            hvx_vmem(سر->v_مفرد_f16 + t * S_v + j) = hvx_vec_f32_مقصد_f16(vp0, vp1);
        }
    }

    hmx_interleave_ستون_تعداد_مقصد_tiles(سر->v_مفرد_ستون_tiles, سر->v_مفرد_f16, 64, S_v, S_v, 2, 0, 64);

    gdn_unpack_64x64_tiles_مقصد_vectors(سر->صفیں_qk, سر->qk_tiles);
    برائے (uint32_t t = 0; t < 64; ++t) {
        HVX_سمتیہ v_decay_a = hvx_vmem(سر->decay_a + t * 64);
        سر->صفیں_a[t]      = hvx_vec_mul_f16_f16(سر->صفیں_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_مقصد_tiles(سر->a_صف_tiles, سر->صفیں_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

ساکن خالی gdn_hvx_phase4_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_مقصد_f16(سر->delta_f16, سر->delta_tiles, S_v);
    hmx_interleave_ستون_تعداد_مقصد_tiles(سر->delta_ستون_tiles, سر->delta_f16, 64, S_v, S_v, 2, 0, 64);

    ثابت uint16_t * decay_آخری = (ثابت uint16_t *) (سر->decay_a + 63 * 64);
    ثابت HVX_سمتیہ vzero      = Q6_V_vzero();

    برائے (uint32_t s = 0; s < 64; ++s) {
        HVX_سمتیہ vs         = Q6_Vh_vsplat_R(decay_آخری[s]);
        HVX_سمتیہ_Pred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
        برائے (uint32_t j = 0; j < S_v; j += 64) {
            HVX_سمتیہ vd   = hvx_vmem(سر->delta_f16 + s * S_v + j);
            HVX_سمتیہ prod = hvx_vec_mul_f16_f16(vd, vs);
            hvx_vmem(سر->d_f16 + s * S_v + j) = Q6_V_vmux_QVV(p_zero, vzero, prod);
        }
    }

    gdn_pack_d_t_صف_tiles(سر->d_صف_tiles, سر->d_f16, S_v, سر->vtcm_m, سر->vtcm_tmp);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

ساکن خالی gdn_hvx_phase5_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_برآمد, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t curr_buf = bctx->curr_buf;
    ثابت uint32_t S_v = bctx->S_v;
    ثابت اعشاری scale = bctx->scale;

    gdn_unpack_64x_S_tiles_مقصد_f32(سر->o_inter_f32, سر->o_inter_tiles, S_v);
    gdn_unpack_64x_S_tiles_مقصد_f32(سر->o_intra_f32, سر->o_intra_tiles, S_v);

    HVX_سمتیہ vscale = hvx_vec_splat_f32(scale);
    برائے (uint32_t j = 0; j < 64 * S_v / 32; ++j) {
        HVX_سمتیہ vi = hvx_vmem(سر->o_inter_f32 + j * 32);
        HVX_سمتیہ va = hvx_vmem(سر->o_intra_f32 + j * 32);
        hvx_vmem(سر->o_f32[curr_buf] + j * 32) = hvx_vec_mul_f32_f32(hvx_vec_جوڑو_f32_f32(vi, va), vscale);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_برآمد, info);
}

ساکن خالی gdn_hvx_phase6_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    (خالی) n;
    ساخت htp_gdn_batch_context * bctx = (ساخت htp_gdn_batch_context *) ڈیٹا;
    ساخت htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ثابت uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    ساخت htp_gdn_سر_ptrs * سر = &bctx->heads[i];
    ثابت uint32_t S_v = bctx->S_v;
    ثابت uint32_t c = bctx->c;
    ثابت uint32_t n_chunks = bctx->kparams->n_chunks;

    gdn_unpack_Sx_S_tiles_مقصد_f32(سر->s_update_f32, سر->s_update_tiles, S_v);

    HVX_سمتیہ_Alias آخری_lambda;
    آخری_lambda.v = hvx_vmem(سر->lambda_init + 32);
    HVX_سمتیہ v_l_final = hvx_vec_splat_f32(آخری_lambda.fp32[31]);

    برائے (uint32_t j = 0; j < S_v * S_v / 32; ++j) {
        HVX_سمتیہ vs_old = hvx_vmem(سر->s_state + j * 32);
        HVX_سمتیہ vsu    = hvx_vmem(سر->s_update_f32 + j * 32);
        hvx_vmem(سر->s_state + j * 32) = hvx_vec_جوڑو_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
    }

    اگر (c + 1 < n_chunks) {
        gdn_pack_s_ستون_tiles(سر->s_ستون_tiles, سر->s_f16, سر->s_state, S_v);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


ساکن صحیح gated_delta_net_f32_hmx_chunked(
    ساخت htp_ops_context * octx,
    ثابت ساخت htp_gdn_kernel_params * kparams,
    uint32_t صف_آغاز,
    uint32_t nrows
) {
    ثابت ساخت htp_tensor * q         = octx->src[0];
    ثابت ساخت htp_tensor * k         = octx->src[1];
    ثابت ساخت htp_tensor * v         = octx->src[2];
    ثابت ساخت htp_tensor * g         = octx->src[3];
    ثابت ساخت htp_tensor * beta      = octx->src[4];
    ثابت ساخت htp_tensor * state     = octx->src[5];
    ثابت ساخت htp_tensor * dst       = octx->dst;
    ثابت ساخت htp_tensor * dst_cache = octx->dsts[1];

    ثابت uint32_t S_v        = kparams->S_v;
    ثابت uint32_t H          = kparams->H;
    ثابت uint32_t n_علامات   = kparams->n_علامات;
    ثابت اعشاری    scale      = kparams->scale;
    ثابت uint32_t chunk_سائز = kparams->chunk_سائز;
    ثابت uint32_t n_chunks   = kparams->n_chunks;
    ثابت uint32_t n_sv_tiles = S_v / 32;

    ساخت htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_سائز, kparams->n_heads_batch, kparams->n_دھاگے, kparams->pipeline != 0);

    اگر (L.کل_بائٹیں > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت vtcm_بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;

    اعشاری * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L.off_g_raw[0])
    };
    اعشاری * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(اعشاری, vtcm_بنیاد, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_بنیاد, L.off_scales_1);
    hmx_init_عمود_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_قطار_t  hmx_q = octx->ctx->hmx_قطار;
    dma_قطار *  dma_q = octx->ctx->dma[0];
    work_قطار_t wp    = octx->ctx->work_قطار;

    ساخت htp_gdn_سر_ptrs heads[8];
    ساخت htp_gdn_hmx_gemm_task gemm_tasks[8][9];

    uint32_t n_batch = 1;
    برائے (uint32_t r = صف_آغاز; r < صف_آغاز + nrows; r += n_batch) {
        ثابت uint32_t سر_اندر_seq         = fastmodulo(r, H, &kparams->div_H);
        ثابت uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        ثابت uint32_t heads_بایاں_اندر_seq   = H - سر_اندر_seq;
        ثابت uint32_t heads_بایاں_اندر_range = (صف_آغاز + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_بایاں_اندر_seq, heads_بایاں_اندر_range));

        برائے (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_سر_ptrs(&heads[h], &L, vtcm_بنیاد, h, سر_اندر_seq, iv3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_علامات, chunk_سائز);
        }

        ساخت htp_gdn_batch_context bctx;
        bctx.heads      = heads;
        bctx.vtcm_g_raw = تہی;
        bctx.vtcm_b_raw = تہی;
        bctx.curr_buf   = 0;
        bctx.c          = 0;
        bctx.n_batch    = n_batch;
        bctx.S_v        = S_v;
        bctx.scale      = scale;
        bctx.octx       = octx;
        bctx.kparams    = kparams;

        برائے (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(heads[h].s_state, heads[h].state_اندر_dma),
                           S_v * حجم(اعشاری), S_v * حجم(اعشاری), S_v * حجم(اعشاری), S_v);
        }
        برائے (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_نکالو(dma_q);
        }

        اگر (n_chunks > 0) {
            work_قطار_تسلسل(wp, gdn_hvx_init_state_کارکن, &bctx, n_batch);

            ثابت uint32_t chunk0_علامات = hex_smin(chunk_سائز, n_علامات);
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                gdn_dma_ڈالو_chunk_inputs(dma_q, heads[h].q_f32[0], heads[h].k_f32[0], heads[h].v_f32[0],
                                          q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                          heads[h].iv3, heads[h].iv1, 0, chunk0_علامات, S_v);
            }
            gdn_dma_ڈالو_chunk_gb(dma_q, vtcm_g_raw[0], vtcm_b_raw[0], g, beta, iv3, سر_اندر_seq, 0, chunk0_علامات, n_batch);
        }

        برائے (uint32_t c = 0; c < n_chunks; ++c) {
            ثابت uint32_t curr_buf = c & 1;
            ثابت uint32_t اگلا_buf = (c + 1) & 1;
            ثابت uint32_t t_chunk  = c * chunk_سائز;

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                dma_قطار_نکالو(dma_q);
                dma_قطار_نکالو(dma_q);
                dma_قطار_نکالو(dma_q);
            }
            dma_قطار_نکالو(dma_q);
            dma_قطار_نکالو(dma_q);

            اگر (c + 1 < n_chunks) {
                ثابت uint32_t اگلا_t_chunk = (c + 1) * chunk_سائز;
                ثابت uint32_t اگلا_علامات_2  = hex_smin(chunk_سائز, n_علامات - اگلا_t_chunk);
                برائے (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_ڈالو_chunk_inputs(dma_q, heads[h].q_f32[اگلا_buf], heads[h].k_f32[اگلا_buf], heads[h].v_f32[اگلا_buf],
                                              q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                              heads[h].iv3, heads[h].iv1, اگلا_t_chunk, اگلا_علامات_2, S_v);
                }
                gdn_dma_ڈالو_chunk_gb(dma_q, vtcm_g_raw[اگلا_buf], vtcm_b_raw[اگلا_buf],
                                      g, beta, iv3, سر_اندر_seq, اگلا_t_chunk, اگلا_علامات_2, n_batch);
            }

            اگر (c > 0) {
                برائے (uint32_t h = 0; h < n_batch; ++h) {
                    dma_قطار_نکالو(dma_q);
                }
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase1a_کارکن, &bctx, n_batch);

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][0], heads[h].k_صف_tiles, heads[h].k_ستون_tiles, heads[h].kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][2], heads[h].k_مفرد_صف_tiles, heads[h].s_ستون_tiles, heads[h].v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase1b_کارکن, &bctx, n_batch);

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][1], heads[h].q_صف_tiles, heads[h].k_ستون_tiles, heads[h].qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][3], heads[h].q_مفرد_صف_tiles, heads[h].s_ستون_tiles, heads[h].o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase2_کارکن, &bctx, n_batch);

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][7],
                    (__fp16 *) heads[h].vtcm_m,
                    heads[h].inv_صف_tiles + 0 * HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    1, 1, 1, vtcm_scales_1
                );
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][8],
                    (__fp16 *) heads[h].vtcm_m + HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    heads[h].inv_صف_tiles + 2 * HMX_FP16_TILE_N_ELMS,
                    1, 1, 1, vtcm_scales_1
                );
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase3_کارکن, &bctx, n_batch);

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][4], heads[h].inv_صف_tiles, heads[h].v_مفرد_ستون_tiles, heads[h].delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase4_کارکن, &bctx, n_batch);

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][5], heads[h].a_صف_tiles, heads[h].delta_ستون_tiles, heads[h].o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ڈالو_hmx_gemm_task(hmx_q, &gemm_tasks[h][6], heads[h].d_صف_tiles, heads[h].k_ستون_tiles_64x128, heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);
            }

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase5_کارکن, &bctx, n_batch);

            ثابت uint32_t valid_علامات = hex_smin(chunk_سائز, n_علامات - t_chunk);
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                ثابت dma_addr_t attn_chunk_dma = dst->ڈیٹا +
                    ((uint64_t) heads[h].iv3 * n_علامات * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * حجم(اعشاری);
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * حجم(اعشاری), S_v * حجم(اعشاری), valid_علامات);
            }

            برائے (uint32_t h = 0; h < n_batch; ++h) {
                hmx_قطار_نکالو(hmx_q);
            }

            work_قطار_تسلسل(wp, gdn_hvx_phase6_کارکن, &bctx, n_batch);
        }

        اگر (n_chunks > 0) {
            برائے (uint32_t h = 0; h < n_batch; ++h) {
                dma_قطار_نکالو(dma_q);
            }
        }

        برائے (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(heads[h].state_برآمد_dma, heads[h].s_state),
                           S_v * حجم(اعشاری), S_v * حجم(اعشاری), S_v * حجم(اعشاری), S_v);
        }
        برائے (uint32_t h = 0; h < n_batch; ++h) {
            dma_قطار_نکالو(dma_q);
        }
    }

    dma_قطار_flush(dma_q);
    واپس HTP_حالت_OK;
}

صحیح op_gated_delta_net(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * q     = octx->src[0];
    ثابت ساخت htp_tensor * k     = octx->src[1];
    ثابت ساخت htp_tensor * v     = octx->src[2];
    ثابت ساخت htp_tensor * g     = octx->src[3];
    ثابت ساخت htp_tensor * beta  = octx->src[4];
    ثابت ساخت htp_tensor * state = octx->src[5];
    ثابت ساخت htp_tensor * dst   = octx->dst;

    اگر (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t S_v      = v->ne[0];
    ثابت uint32_t H        = v->ne[1];
    ثابت uint32_t n_علامات = v->ne[2];
    ثابت uint32_t n_seqs   = v->ne[3];
    ثابت uint32_t K        = octx->op_params[0];

    اگر (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_علامات == 0 || n_seqs == 0) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    اگر ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    اگر (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_علامات || k->ne[2] != n_علامات || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    اگر (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    اگر (dst->ne[0] != S_v * H || dst->ne[1] != n_علامات * n_seqs + S_v * n_seqs * K) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    برائے (صحیح i = 0; i < 5; i++) {
        اگر (htp_tensor_is_extended(octx->src[i])) {
            واپس HTP_حالت_NO_SUPPORT;
        }
    }
    اگر (htp_tensor_is_extended(octx->dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    اگر (octx->dsts[1]) {
        ثابت ساخت htp_tensor * dst_cache = octx->dsts[1];
        اگر (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            واپس HTP_حالت_NO_SUPPORT;
        }
    }

    ثابت ساخت htp_gdn_kernel_params * kparams = (ثابت ساخت htp_gdn_kernel_params *) octx->kernel_params;
    ساخت htp_gdn_kernel_params kparams_local;
    اگر (!kparams || kparams->S_v == 0) {
        ثابت uint32_t rq3 = n_seqs / q->ne[3];
        ثابت uint32_t rk3 = n_seqs / k->ne[3];
        ثابت uint32_t کل_صفیں = H * n_seqs;
        uint32_t n_دھاگے = (کل_صفیں < octx->n_دھاگے) ? کل_صفیں : octx->n_دھاگے;
        اگر (n_دھاگے == 0) {
            n_دھاگے = 1;
        }

        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_دھاگے           = n_دھاگے;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_علامات            = n_علامات;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.کل_صفیں          = کل_صفیں;
        kparams_local.صفیں_فی_thread     = (کل_صفیں + n_دھاگے - 1) / n_دھاگے;
        ثابت bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_علامات >= HTP_GDN_MIN_علامات) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        ساخت htp_gdn_hmx_vtcm_layout hmx_layout_local;
        ساخت htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        اگر (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_سائز, کل_صفیں, octx->ctx->vtcm_سائز, n_دھاگے, true, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_سائز      = HTP_GDN_CHUNK_سائز;
            kparams_local.n_chunks        = (n_علامات + HTP_GDN_CHUNK_سائز - 1) / HTP_GDN_CHUNK_سائز;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_سائز       = (uint32_t) hmx_layout_local.کل_بائٹیں;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_بائٹیں;
            kparams_local.vtcm_فی_thread = (uint32_t) (hmx_layout_local.کل_بائٹیں / (n_دھاگے > 0 ? n_دھاگے : 1));
        } ورنہ {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_دھاگے);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_فی_thread = (uint32_t) hvx_layout_local.بائٹیں_فی_thread;
            kparams_local.vtcm_سائز       = (uint32_t) hvx_layout_local.کل_بائٹیں;
        }
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((اعشاری) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / حجم(اعشاری));
        kparams_local.state_سائز_فی_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_اقدار(H);
        kparams_local.div_q1        = init_fastdiv_اقدار(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_اقدار(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_اقدار(rq3);
        kparams_local.div_rk3       = init_fastdiv_اقدار(rk3);
        kparams_local.div_n_دھاگے = init_fastdiv_اقدار(n_دھاگے);

        kparams = &kparams_local;
    }

    ثابت uint32_t کل_صفیں = kparams->کل_صفیں;
    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت bool can_بانٹو = htp_tensor_mdev_ڈیٹا_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            کل_صفیں,
            can_بانٹو ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.گنتی,
            &octx->ctx->mdev.گنتی_div
        );
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    } ورنہ اگر (octx->op_params[1] != 0) {
        صف_آغاز = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        واپس gated_delta_net_f32_hmx_chunked(octx, kparams, صف_آغاز, nrows);
    }

    ثابت uint32_t n_دھاگے = (nrows < kparams->n_دھاگے) ? nrows : kparams->n_دھاگے;

    ساخت htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.صف_آغاز = صف_آغاز;
    gctx.nrows     = nrows;
    gctx.vtcm_بنیاد = octx->ctx->vtcm_بنیاد;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_دھاگے);

    اگر (gctx.layout.کل_بائٹیں > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    FARF(بلند_2, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.کل_بائٹیں, n_دھاگے);

    اگر (n_علامات == 1) {
        work_قطار_تسلسل(octx->ctx->work_قطار, gated_delta_net_f32_tg_thread, &gctx, n_دھاگے);
    } ورنہ {
        work_قطار_تسلسل(octx->ctx->work_قطار, gated_delta_net_f32_pp_thread, &gctx, n_دھاگے);
    }

    واپس HTP_حالت_OK;
}
