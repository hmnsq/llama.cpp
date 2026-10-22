#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-reduce.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-exp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-fa-kernels.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gated-delta-net-ops.h"

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MIN
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MIN(a, b) ((a) < (b) ? (a) : (b))
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_gdn_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams;
    ਬਣਤਰ htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_ਅਧਾਰ;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t nrows;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_mul_dot_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ mul, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc = Q6_V_vzero();
    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vd   = hvx_vmemu(dst + i * epv);
        HVX_ਸਦਿਸ਼ vm   = mul[i];
        HVX_ਸਦਿਸ਼ vdot = dot[i];
        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = ਬਾਹਰਲਾ;
        acc = hvx_vec_ਜੋੜੋ_f32_f32(acc, hvx_vec_mul_f32_f32(ਬਾਹਰਲਾ, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vm = mul[nvec];
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਬਾਹਰਲਾ);
        acc = hvx_vec_ਜੋੜੋ_f32_f32(acc, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(ਬਾਹਰਲਾ, vdot), zero));
    }

    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32(acc);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_mul_scalar_dot_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst, HVX_ਸਦਿਸ਼ vmul, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc = Q6_V_vzero();
    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vd   = hvx_vmemu(dst + i * epv);
        HVX_ਸਦਿਸ਼ vdot = dot[i];
        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = ਬਾਹਰਲਾ;
        acc = hvx_vec_ਜੋੜੋ_f32_f32(acc, hvx_vec_mul_f32_f32(ਬਾਹਰਲਾ, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਬਾਹਰਲਾ);
        acc = hvx_vec_ਜੋੜੋ_f32_f32(acc, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(ਬਾਹਰਲਾ, vdot), zero));
    }

    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32(acc);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_ਜੋੜੋ_scaled_dot_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ src,
        HVX_ਸਦਿਸ਼ vscale, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc = Q6_V_vzero();
    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vd   = hvx_vmemu(dst + i * epv);
        HVX_ਸਦਿਸ਼ vs   = src[i];
        HVX_ਸਦਿਸ਼ vdot = dot[i];
        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ  = hvx_vec_ਜੋੜੋ_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = ਬਾਹਰਲਾ;
        acc = hvx_vec_ਜੋੜੋ_f32_f32(acc, hvx_vec_mul_f32_f32(ਬਾਹਰਲਾ, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vs   = src[nvec];
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਬਾਹਰਲਾ);
        acc = hvx_vec_ਜੋੜੋ_f32_f32(acc, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(ਬਾਹਰਲਾ, vdot), zero));
    }

    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32(acc);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_mul_dot4_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst1,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst2, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst3,
        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ mul, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc1 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc2 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc3 = Q6_V_vzero();

    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vm   = mul[i];
        HVX_ਸਦਿਸ਼ vdot = dot[i];

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vm = mul[nvec];
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out3);

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ਸਦਿਸ਼_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32x4(acc);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_mul_scalar_dot4_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst1,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst2, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst3,
        HVX_ਸਦਿਸ਼ vmul, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc1 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc2 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc3 = Q6_V_vzero();

    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vdot = dot[i];

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out3);

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ਸਦਿਸ਼_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32x4(acc);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_ਜੋੜੋ_scaled_dot4_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst1,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst2, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst3,
        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ scale,
        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc1 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc2 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc3 = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ scale0 = hvx_vec_splat_f32(scale[0]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale1 = hvx_vec_splat_f32(scale[1]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale2 = hvx_vec_splat_f32(scale[2]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale3 = hvx_vec_splat_f32(scale[3]);

    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vs   = src[i];
        HVX_ਸਦਿਸ਼ vdot = dot[i];

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vs   = src[nvec];
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out3);

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ਸਦਿਸ਼_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32x4(acc);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_mul_dot8_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst1,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst2, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst3, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst4,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst5, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst6, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst7,
        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ mul, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc1 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc2 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc3 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc4 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc5 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc6 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc7 = Q6_V_vzero();

    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vm   = mul[i];
        HVX_ਸਦਿਸ਼ vdot = dot[i];

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_ਸਦਿਸ਼ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_ਜੋੜੋ_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_ਜੋੜੋ_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_ਜੋੜੋ_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_ਜੋੜੋ_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vm = mul[nvec];
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_ਸਦਿਸ਼ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_ਸਦਿਸ਼ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_ਸਦਿਸ਼ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_ਸਦਿਸ਼ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out3);
        hvx_vec_store_u(dst4 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out4);
        hvx_vec_store_u(dst5 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out5);
        hvx_vec_store_u(dst6 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out6);
        hvx_vec_store_u(dst7 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out7);

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_ਜੋੜੋ_f32_f32(acc4, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_ਜੋੜੋ_f32_f32(acc5, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_ਜੋੜੋ_f32_f32(acc6, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_ਜੋੜੋ_f32_f32(acc7, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ਸਦਿਸ਼_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ਸਦਿਸ਼_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ਸਦਿਸ਼ rA = hvx_vec_reduce_ਜੋੜ_f32x4(accA);
    HVX_ਸਦਿਸ਼ rB = hvx_vec_reduce_ਜੋੜ_f32x4(accB);
    HVX_ਸਦਿਸ਼_Pred q16 = Q6_Q_vsetq2_R(16);
    ਵਾਪਸ Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_mul_scalar_dot8_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst1,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst2, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst3, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst4,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst5, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst6, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst7,
        HVX_ਸਦਿਸ਼ vmul, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc1 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc2 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc3 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc4 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc5 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc6 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc7 = Q6_V_vzero();

    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vdot = dot[i];

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_ਸਦਿਸ਼ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_ਜੋੜੋ_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_ਜੋੜੋ_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_ਜੋੜੋ_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_ਜੋੜੋ_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_ਸਦਿਸ਼ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_ਸਦਿਸ਼ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_ਸਦਿਸ਼ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_ਸਦਿਸ਼ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out3);
        hvx_vec_store_u(dst4 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out4);
        hvx_vec_store_u(dst5 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out5);
        hvx_vec_store_u(dst6 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out6);
        hvx_vec_store_u(dst7 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out7);

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_ਜੋੜੋ_f32_f32(acc4, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_ਜੋੜੋ_f32_f32(acc5, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_ਜੋੜੋ_f32_f32(acc6, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_ਜੋੜੋ_f32_f32(acc7, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ਸਦਿਸ਼_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ਸਦਿਸ਼_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ਸਦਿਸ਼ rA = hvx_vec_reduce_ਜੋੜ_f32x4(accA);
    HVX_ਸਦਿਸ਼ rB = hvx_vec_reduce_ਜੋੜ_f32x4(accB);
    HVX_ਸਦਿਸ਼_Pred q16 = Q6_Q_vsetq2_R(16);
    ਵਾਪਸ Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ gdn_ਜੋੜੋ_scaled_dot8_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst1,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst2, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst3, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst4,
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst5, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst6, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst7,
        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ scale,
        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ dot, uint32_t n) {
    HVX_ਸਦਿਸ਼ acc0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc1 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc2 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc3 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc4 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc5 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc6 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ acc7 = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ scale0 = hvx_vec_splat_f32(scale[0]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale1 = hvx_vec_splat_f32(scale[1]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale2 = hvx_vec_splat_f32(scale[2]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale3 = hvx_vec_splat_f32(scale[3]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale4 = hvx_vec_splat_f32(scale[4]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale5 = hvx_vec_splat_f32(scale[5]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale6 = hvx_vec_splat_f32(scale[6]);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale7 = hvx_vec_splat_f32(scale[7]);

    ਅਚਲ uint32_t epv = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;
    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ਸਦਿਸ਼ vs   = src[i];
        HVX_ਸਦਿਸ਼ vdot = dot[i];

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_ਸਦਿਸ਼ out4 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_ਸਦਿਸ਼ out5 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_ਸਦਿਸ਼ out6 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_ਸਦਿਸ਼ out7 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_ਜੋੜੋ_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_ਜੋੜੋ_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_ਜੋੜੋ_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_ਜੋੜੋ_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ਜੇ (nloe) {
        ਅਚਲ uint32_t off = nvec * epv;
        HVX_ਸਦਿਸ਼ vs   = src[nvec];
        HVX_ਸਦਿਸ਼ vdot = dot[nvec];
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

        HVX_ਸਦਿਸ਼ out0 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ਸਦਿਸ਼ out1 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ਸਦਿਸ਼ out2 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ਸਦਿਸ਼ out3 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_ਸਦਿਸ਼ out4 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_ਸਦਿਸ਼ out5 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_ਸਦਿਸ਼ out6 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_ਸਦਿਸ਼ out7 = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out3);
        hvx_vec_store_u(dst4 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out4);
        hvx_vec_store_u(dst5 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out5);
        hvx_vec_store_u(dst6 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out6);
        hvx_vec_store_u(dst7 + off, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), out7);

        acc0 = hvx_vec_ਜੋੜੋ_f32_f32(acc0, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ਜੋੜੋ_f32_f32(acc1, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ਜੋੜੋ_f32_f32(acc2, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ਜੋੜੋ_f32_f32(acc3, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_ਜੋੜੋ_f32_f32(acc4, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_ਜੋੜੋ_f32_f32(acc5, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_ਜੋੜੋ_f32_f32(acc6, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_ਜੋੜੋ_f32_f32(acc7, Q6_V_vmux_QVV(ਨਕਾਬ, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ਸਦਿਸ਼_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ਸਦਿਸ਼_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ਸਦਿਸ਼ rA = hvx_vec_reduce_ਜੋੜ_f32x4(accA);
    HVX_ਸਦਿਸ਼ rB = hvx_vec_reduce_ਜੋੜ_f32x4(accB);
    HVX_ਸਦਿਸ਼_Pred q16 = Q6_Q_vsetq2_R(16);
    ਵਾਪਸ Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_ਕਦਮ_kda_f32(
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s_work,
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ attn_ਬਾਹਰਲਾ,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ q_t,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ k_t,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ v_t,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ g_t,
    ਦਸ਼ਮਲਵ beta_val,
    ਦਸ਼ਮਲਵ scale,
    uint32_t S_v
) {
    ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = S_v / epv;
    ਅਚਲ uint32_t nloe = S_v % epv;

    HVX_ਸਦਿਸ਼ vq[4];
    HVX_ਸਦਿਸ਼ vk[4];
    HVX_ਸਦਿਸ਼ vg[4];

    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    ਜੇ (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
    }

    ਅਚਲ HVX_ਸਦਿਸ਼ vbeta  = hvx_vec_splat_f32(beta_val);
    ਅਚਲ HVX_ਸਦਿਸ਼ vscale = hvx_vec_splat_f32(scale);

    ਦਸ਼ਮਲਵ delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    ਲਈ (; j + 8 <= S_v; j += 8) {
        ਦਸ਼ਮਲਵ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ਦਸ਼ਮਲਵ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ਦਸ਼ਮਲਵ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ਦਸ਼ਮਲਵ * row3 = s_work + (uint64_t) (j + 3) * S_v;
        ਦਸ਼ਮਲਵ * row4 = s_work + (uint64_t) (j + 4) * S_v;
        ਦਸ਼ਮਲਵ * row5 = s_work + (uint64_t) (j + 5) * S_v;
        ਦਸ਼ਮਲਵ * row6 = s_work + (uint64_t) (j + 6) * S_v;
        ਦਸ਼ਮਲਵ * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_ਸਦਿਸ਼ vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_ਸਦਿਸ਼ vv_t   = hvx_vmemu(v_t + j);
        HVX_ਸਦਿਸ਼ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ਸਦਿਸ਼ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vdelta);

        HVX_ਸਦਿਸ਼ vattn = gdn_ਜੋੜੋ_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_ਸਦਿਸ਼ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ਬਾਹਰਲਾ + j, 8 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res_attn);
    }
    ਲਈ (; j + 4 <= S_v; j += 4) {
        ਦਸ਼ਮਲਵ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ਦਸ਼ਮਲਵ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ਦਸ਼ਮਲਵ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ਦਸ਼ਮਲਵ * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_ਸਦਿਸ਼ vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_ਸਦਿਸ਼ vv_t   = hvx_vmemu(v_t + j);
        HVX_ਸਦਿਸ਼ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ਸਦਿਸ਼ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vdelta);

        HVX_ਸਦਿਸ਼ vattn = gdn_ਜੋੜੋ_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_ਸਦਿਸ਼ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ਬਾਹਰਲਾ + j, 4 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res_attn);
    }
    ਲਈ (; j < S_v; ++j) {
        ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = s_work + (uint64_t) j * S_v;
        HVX_ਸਦਿਸ਼ vsum = gdn_mul_dot_f32(ਪੰਕਤੀ, vg, vk, S_v);
        HVX_ਸਦਿਸ਼ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_ਸਦਿਸ਼ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_ਸਦਿਸ਼ vres = gdn_ਜੋੜੋ_scaled_dot_f32(ਪੰਕਤੀ, vk, vdj, vq, S_v);
        attn_ਬਾਹਰਲਾ[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_ਕਦਮ_scalar_f32(
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s_work,
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ attn_ਬਾਹਰਲਾ,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ q_t,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ k_t,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ v_t,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ g_t,
    ਦਸ਼ਮਲਵ beta_val,
    ਦਸ਼ਮਲਵ scale,
    uint32_t S_v
) {
    ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = S_v / epv;
    ਅਚਲ uint32_t nloe = S_v % epv;

    HVX_ਸਦਿਸ਼ vq[4];
    HVX_ਸਦਿਸ਼ vk[4];

    ਲਈ (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    ਜੇ (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    ਅਚਲ HVX_ਸਦਿਸ਼ vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    ਅਚਲ HVX_ਸਦਿਸ਼ vbeta  = hvx_vec_splat_f32(beta_val);
    ਅਚਲ HVX_ਸਦਿਸ਼ vscale = hvx_vec_splat_f32(scale);

    ਦਸ਼ਮਲਵ delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    ਲਈ (; j + 8 <= S_v; j += 8) {
        ਦਸ਼ਮਲਵ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ਦਸ਼ਮਲਵ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ਦਸ਼ਮਲਵ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ਦਸ਼ਮਲਵ * row3 = s_work + (uint64_t) (j + 3) * S_v;
        ਦਸ਼ਮਲਵ * row4 = s_work + (uint64_t) (j + 4) * S_v;
        ਦਸ਼ਮਲਵ * row5 = s_work + (uint64_t) (j + 5) * S_v;
        ਦਸ਼ਮਲਵ * row6 = s_work + (uint64_t) (j + 6) * S_v;
        ਦਸ਼ਮਲਵ * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_ਸਦਿਸ਼ vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_ਸਦਿਸ਼ vv_t   = hvx_vmemu(v_t + j);
        HVX_ਸਦਿਸ਼ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ਸਦਿਸ਼ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vdelta);

        HVX_ਸਦਿਸ਼ vattn = gdn_ਜੋੜੋ_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_ਸਦਿਸ਼ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ਬਾਹਰਲਾ + j, 8 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res_attn);
    }
    ਲਈ (; j + 4 <= S_v; j += 4) {
        ਦਸ਼ਮਲਵ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ਦਸ਼ਮਲਵ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ਦਸ਼ਮਲਵ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ਦਸ਼ਮਲਵ * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_ਸਦਿਸ਼ vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_ਸਦਿਸ਼ vv_t   = hvx_vmemu(v_t + j);
        HVX_ਸਦਿਸ਼ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ਸਦਿਸ਼ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vdelta);

        HVX_ਸਦਿਸ਼ vattn = gdn_ਜੋੜੋ_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_ਸਦਿਸ਼ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ਬਾਹਰਲਾ + j, 4 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res_attn);
    }
    ਲਈ (; j < S_v; ++j) {
        ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = s_work + (uint64_t) j * S_v;
        HVX_ਸਦਿਸ਼ vsum = gdn_mul_scalar_dot_f32(ਪੰਕਤੀ, vgate, vk, S_v);
        HVX_ਸਦਿਸ਼ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_ਸਦਿਸ਼ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_ਸਦਿਸ਼ vres = gdn_ਜੋੜੋ_scaled_dot_f32(ਪੰਕਤੀ, vk, vdj, vq, S_v);
        attn_ਬਾਹਰਲਾ[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ਸਥਿਰ ਖਾਲੀ gated_delta_net_f32_pp_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_gdn_context * gctx = (ਬਣਤਰ htp_gdn_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = gctx->octx;
    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams = gctx->kparams;

    ਅਚਲ ਬਣਤਰ htp_tensor * q     = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k     = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v     = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * g     = octx->src[3];
    ਅਚਲ ਬਣਤਰ htp_tensor * beta  = octx->src[4];
    ਅਚਲ ਬਣਤਰ htp_tensor * state = octx->src[5];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst   = octx->dst;

    ਅਚਲ uint32_t S_v      = kparams->S_v;
    ਅਚਲ uint32_t H        = kparams->H;
    ਅਚਲ uint32_t n_ਟੋਕਨ = kparams->n_ਟੋਕਨ;
    ਅਚਲ uint32_t n_seqs   = kparams->n_seqs;
    ਅਚਲ uint32_t K        = kparams->K;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ  = gctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + gctx->nrows;

    ਜੇ (ith >= gctx->nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst_cache = octx->dsts[1];
    ਅਚਲ ਦਸ਼ਮਲਵ scale = kparams->scale;
    ਦਸ਼ਮਲਵ * dst_ਅਧਾਰ       = (ਦਸ਼ਮਲਵ *) (uintptr_t) dst->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * state_ਬਾਹਰਲਾ_ਅਧਾਰ = dst_cache ? (ਦਸ਼ਮਲਵ *) (uintptr_t) dst_cache->ਅੰਕੜੇ : (dst_ਅਧਾਰ + S_v * H * n_ਟੋਕਨ * n_seqs);

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    ਅਚਲ ਬਣਤਰ htp_gdn_vtcm_layout * layout = &gctx->layout;
    ਦਸ਼ਮਲਵ * s_work[2];
    s_work[0] = (ਦਸ਼ਮਲਵ *) (gctx->vtcm_ਅਧਾਰ + layout->ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_H   = &kparams->div_H;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_q1  = &kparams->div_q1;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_k1  = &kparams->div_k1;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_rq3 = &kparams->div_rq3;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_rk3 = &kparams->div_rk3;

    ਅਚਲ uint32_t state_seq_stride = kparams->state_seq_stride;
    ਅਚਲ uint64_t state_ਮਾਪ_ਪ੍ਰਤੀ_snap = (uint64_t) kparams->state_ਮਾਪ_ਪ੍ਰਤੀ_snap;
    ਅਚਲ dma_addr_t state_ਬਾਹਰਲਾ_dma_ਅਧਾਰ = dst_cache ? dst_cache->ਅੰਕੜੇ : (dst->ਅੰਕੜੇ + S_v * H * n_ਟੋਕਨ * n_seqs * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    uint32_t ir_prefetch = gctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith;
    ਪੂਰਨ spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    ਲਈ (ਪੂਰਨ ਕਦਮ = 0; ਕਦਮ < 2 && ir_prefetch < ਪੰਕਤੀ_ਸਿਰਾ; ਕਦਮ++) {
        ਅਚਲ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ਅਚਲ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_ਵਿੱਚ  = state->ਅੰਕੜੇ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        dma_addr_t ps_ਬਾਹਰਲਾ = state_ਬਾਹਰਲਾ_dma_ਅਧਾਰ + ((uint64_t) piv3 * H + piv1) * S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        // Push dummy write-back
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ps_ਬਾਹਰਲਾ, s_work[spad_idx]),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 0);

        // Push fetch
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_work[spad_idx], ps_ਵਿੱਚ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਪੂਰਨ curr_spad_idx = 0;
    ਲਈ (uint32_t ir = gctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith; ir < ਪੰਕਤੀ_ਸਿਰਾ; ir += nth) {
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

        ਦਸ਼ਮਲਵ * s_work_curr = s_work[curr_spad_idx];

        ਅਚਲ uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ਅਚਲ uint32_t iv3 = fastdiv(ir, fd_H);

        ਅਚਲ uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ਅਚਲ uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ਅਚਲ uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ਅਚਲ uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_ਬਾਹਰਲਾ  = state_ਬਾਹਰਲਾ_dma_ਅਧਾਰ + ((uint64_t) iv3 * H + iv1) * S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਦਸ਼ਮਲਵ * attn_ਅੰਕੜੇ = dst_ਅਧਾਰ + ((uint64_t) iv3 * n_ਟੋਕਨ * H + iv1) * S_v;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ਲਈ (uint32_t t = 0; t < n_ਟੋਕਨ; ++t) {
            ਅਚਲ ਦਸ਼ਮਲਵ * q_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) q->ਅੰਕੜੇ +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            ਅਚਲ ਦਸ਼ਮਲਵ * k_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) k->ਅੰਕੜੇ +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            ਅਚਲ ਦਸ਼ਮਲਵ * v_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) v->ਅੰਕੜੇ +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            ਅਚਲ ਦਸ਼ਮਲਵ * g_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) g->ਅੰਕੜੇ +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            ਅਚਲ ਦਸ਼ਮਲਵ beta_val = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) beta->ਅੰਕੜੇ +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            ਜੇ (kparams->kda) {
                gdn_ਕਦਮ_kda_f32(s_work_curr, attn_ਅੰਕੜੇ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } ਨਹੀਂਤਾਂ {
                gdn_ਕਦਮ_scalar_f32(s_work_curr, attn_ਅੰਕੜੇ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            ਜੇ (K > 1) {
                ਅਚਲ int64_t ਨਿਸ਼ਾਨਾ_ਥਾਂ = (int64_t) n_ਟੋਕਨ - 1 - (int64_t) t;
                ਜੇ (ਨਿਸ਼ਾਨਾ_ਥਾਂ > 0 && ਨਿਸ਼ਾਨਾ_ਥਾਂ < (int64_t) K) {
                    ਦਸ਼ਮਲਵ * curr_state_o = state_ਬਾਹਰਲਾ_ਅਧਾਰ + (uint64_t) ਨਿਸ਼ਾਨਾ_ਥਾਂ * state_ਮਾਪ_ਪ੍ਰਤੀ_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (ਅਚਲ uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_ਅੰਕੜੇ += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_ਬਾਹਰਲਾ, s_work_curr),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);

        // Prefetch next block (if any)
        ਜੇ (ir_prefetch < ਪੰਕਤੀ_ਸਿਰਾ) {
            ਅਚਲ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ਅਚਲ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_ਵਿੱਚ = state->ਅੰਕੜੇ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_work[spad_idx], ps_ਵਿੱਚ),
                           S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                           S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਖਾਲੀ gated_delta_net_f32_tg_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_gdn_context * gctx = (ਬਣਤਰ htp_gdn_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = gctx->octx;
    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams = gctx->kparams;

    ਅਚਲ ਬਣਤਰ htp_tensor * q     = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k     = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v     = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * g     = octx->src[3];
    ਅਚਲ ਬਣਤਰ htp_tensor * beta  = octx->src[4];
    ਅਚਲ ਬਣਤਰ htp_tensor * state = octx->src[5];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst   = octx->dst;

    ਅਚਲ uint32_t S_v      = kparams->S_v;
    ਅਚਲ uint32_t H        = kparams->H;
    ਅਚਲ uint32_t n_seqs   = kparams->n_seqs;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ  = gctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + gctx->nrows;

    ਜੇ (ith >= gctx->nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst_cache = octx->dsts[1];
    ਅਚਲ ਦਸ਼ਮਲਵ scale = kparams->scale;
    ਦਸ਼ਮਲਵ * dst_ਅਧਾਰ  = (ਦਸ਼ਮਲਵ *) (uintptr_t) dst->ਅੰਕੜੇ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    ਅਚਲ ਬਣਤਰ htp_gdn_vtcm_layout * layout = &gctx->layout;
    ਦਸ਼ਮਲਵ * s_work[2];
    s_work[0] = (ਦਸ਼ਮਲਵ *) (gctx->vtcm_ਅਧਾਰ + layout->ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_H   = &kparams->div_H;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_q1  = &kparams->div_q1;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_k1  = &kparams->div_k1;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_rq3 = &kparams->div_rq3;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * fd_rk3 = &kparams->div_rk3;

    ਅਚਲ uint32_t state_seq_stride = kparams->state_seq_stride;
    ਅਚਲ dma_addr_t state_ਬਾਹਰਲਾ_dma_ਅਧਾਰ = dst_cache ? dst_cache->ਅੰਕੜੇ : (dst->ਅੰਕੜੇ + S_v * H * n_seqs * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    uint32_t ir_prefetch = gctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith;
    ਪੂਰਨ spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    ਲਈ (ਪੂਰਨ ਕਦਮ = 0; ਕਦਮ < 2 && ir_prefetch < ਪੰਕਤੀ_ਸਿਰਾ; ਕਦਮ++) {
        ਅਚਲ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ਅਚਲ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_ਵਿੱਚ  = state->ਅੰਕੜੇ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        dma_addr_t ps_ਬਾਹਰਲਾ = state_ਬਾਹਰਲਾ_dma_ਅਧਾਰ + ((uint64_t) piv3 * H + piv1) * S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        // Push dummy write-back
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ps_ਬਾਹਰਲਾ, s_work[spad_idx]),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 0);

        // Push fetch
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_work[spad_idx], ps_ਵਿੱਚ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਪੂਰਨ curr_spad_idx = 0;
    ਲਈ (uint32_t ir = gctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith; ir < ਪੰਕਤੀ_ਸਿਰਾ; ir += nth) {
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

        ਦਸ਼ਮਲਵ * s_work_curr = s_work[curr_spad_idx];

        ਅਚਲ uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ਅਚਲ uint32_t iv3 = fastdiv(ir, fd_H);

        ਅਚਲ uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ਅਚਲ uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ਅਚਲ uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ਅਚਲ uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_ਬਾਹਰਲਾ  = state_ਬਾਹਰਲਾ_dma_ਅਧਾਰ + ((uint64_t) iv3 * H + iv1) * S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਦਸ਼ਮਲਵ * attn_ਅੰਕੜੇ = dst_ਅਧਾਰ + ((uint64_t) iv3 * H + iv1) * S_v;

        ਅਚਲ ਦਸ਼ਮਲਵ * q_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) q->ਅੰਕੜੇ +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ * k_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) k->ਅੰਕੜੇ +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ * v_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) v->ਅੰਕੜੇ +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ * g_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) g->ਅੰਕੜੇ +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ beta_val = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) beta->ਅੰਕੜੇ +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ਜੇ (kparams->kda) {
            gdn_ਕਦਮ_kda_f32(s_work_curr, attn_ਅੰਕੜੇ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } ਨਹੀਂਤਾਂ {
            gdn_ਕਦਮ_scalar_f32(s_work_curr, attn_ਅੰਕੜੇ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_ਬਾਹਰਲਾ, s_work_curr),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                       S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);

        // Prefetch next block (if any)
        ਜੇ (ir_prefetch < ਪੰਕਤੀ_ਸਿਰਾ) {
            ਅਚਲ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ਅਚਲ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_ਵਿੱਚ = state->ਅੰਕੜੇ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_work[spad_idx], ps_ਵਿੱਚ),
                           S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                           S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਬਣਤਰ htp_gdn_hmx_gemm_task {
    ਅਚਲ __fp16 * ਪੰਕਤੀ_tiles;
    ਅਚਲ __fp16 * ਖਾਨਾ_tiles;
    __fp16 *       ਬਾਹਰਲਾ_tiles;
    uint32_t       n_ਪੰਕਤੀ_tiles;
    uint32_t       n_ਖਾਨਾ_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

ਸਥਿਰ ਖਾਲੀ htp_gdn_hmx_gemm_ਕਾਮਾ(ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_gdn_hmx_gemm_task * task = (ਬਣਤਰ htp_gdn_hmx_gemm_task *) ਅੰਕੜੇ;
    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)task->hmx_scales));

    ਅਚਲ size_t dot_stride = task->dot_stride;
    ਲਈ (uint32_t r = 0; r < task->n_ਪੰਕਤੀ_tiles; ++r) {
        ਅਚਲ __fp16 * r_tiles = task->ਪੰਕਤੀ_tiles + r * dot_stride;
        ਅਚਲ __fp16 * c_tiles = task->ਖਾਨਾ_tiles;
        __fp16 *       o_tile  = task->ਬਾਹਰਲਾ_tiles + r * task->n_ਖਾਨਾ_tiles * HMX_FP16_TILE_N_ELMS;

        ਲਈ (uint32_t c = 0; c < task->n_ਖਾਨਾ_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_gdn_ਧੱਕੋ_hmx_gemm_task(
    hmx_ਕਤਾਰ_t q,
    ਬਣਤਰ htp_gdn_hmx_gemm_task * task,
    ਅਚਲ __fp16 * ਪੰਕਤੀ_tiles,
    ਅਚਲ __fp16 * ਖਾਨਾ_tiles,
    __fp16 * ਬਾਹਰਲਾ_tiles,
    uint32_t n_ਪੰਕਤੀ_tiles,
    uint32_t n_ਖਾਨਾ_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    task->ਪੰਕਤੀ_tiles   = ਪੰਕਤੀ_tiles;
    task->ਖਾਨਾ_tiles   = ਖਾਨਾ_tiles;
    task->ਬਾਹਰਲਾ_tiles   = ਬਾਹਰਲਾ_tiles;
    task->n_ਪੰਕਤੀ_tiles = n_ਪੰਕਤੀ_tiles;
    task->n_ਖਾਨਾ_tiles = n_ਖਾਨਾ_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    task->hmx_scales  = scales;

    hmx_ਕਤਾਰ_ਧੱਕੋ(q, hmx_ਕਤਾਰ_make_desc(htp_gdn_hmx_gemm_ਕਾਮਾ, task));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_unpack_64x64_tiles_ਮੰਜ਼ਿਲ_vectors(
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ ਪੰਕਤੀਆਂ,
    ਅਚਲ __fp16 * ਸੀਮਿਤ tiles
) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * t00 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    ਅਚਲ HVX_ਸਦਿਸ਼ * t01 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    ਅਚਲ HVX_ਸਦਿਸ਼ * t10 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    ਅਚਲ HVX_ਸਦਿਸ਼ * t11 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        HVX_ਸਦਿਸ਼_Pair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        ਪੰਕਤੀਆਂ[2 * r + 0] = Q6_V_lo_W(vp0);
        ਪੰਕਤੀਆਂ[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_ਸਦਿਸ਼_Pair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        ਪੰਕਤੀਆਂ[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        ਪੰਕਤੀਆਂ[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_pack_64x64_vectors_ਮੰਜ਼ਿਲ_tiles(
    __fp16 * ਸੀਮਿਤ tiles,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ ਪੰਕਤੀਆਂ
) {
    HVX_ਸਦਿਸ਼ * t00 = (HVX_ਸਦਿਸ਼ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_ਸਦਿਸ਼ * t01 = (HVX_ਸਦਿਸ਼ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_ਸਦਿਸ਼ * t10 = (HVX_ਸਦਿਸ਼ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_ਸਦਿਸ਼ * t11 = (HVX_ਸਦਿਸ਼ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        HVX_ਸਦਿਸ਼_Pair vp0 = Q6_W_vshuff_VVR(ਪੰਕਤੀਆਂ[2 * r + 1], ਪੰਕਤੀਆਂ[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_ਸਦਿਸ਼_Pair vp1 = Q6_W_vshuff_VVR(ਪੰਕਤੀਆਂ[32 + 2 * r + 1], ਪੰਕਤੀਆਂ[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_unpack_64x_S_tiles_ਮੰਜ਼ਿਲ_f32(
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_f32,
    ਅਚਲ __fp16 * ਸੀਮਿਤ tiles,
    uint32_t S_v
) {
    ਅਚਲ uint32_t n_ਖਾਨਾ_tiles = S_v / 32;
    ਲਈ (uint32_t r0 = 0; r0 < 2; ++r0) {
        ਲਈ (uint32_t d = 0; d < S_v / 64; ++d) {
            ਅਚਲ HVX_ਸਦਿਸ਼ * t0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + (r0 * n_ਖਾਨਾ_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ਅਚਲ HVX_ਸਦਿਸ਼ * t1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + (r0 * n_ਖਾਨਾ_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            ਲਈ (uint32_t r = 0; r < 16; ++r) {
                HVX_ਸਦਿਸ਼_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_ਸਦਿਸ਼_Pair p0 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(Q6_V_lo_W(vp01));
                HVX_ਸਦਿਸ਼_Pair p1 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(Q6_V_hi_W(vp01));

                ਦਸ਼ਮਲਵ * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                ਦਸ਼ਮਲਵ * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_unpack_64x_S_tiles_ਮੰਜ਼ਿਲ_f16(
    __fp16 * ਸੀਮਿਤ dst_f16,
    ਅਚਲ __fp16 * ਸੀਮਿਤ tiles,
    uint32_t S_v
) {
    ਅਚਲ uint32_t n_ਖਾਨਾ_tiles = S_v / 32;
    ਲਈ (uint32_t r0 = 0; r0 < 2; ++r0) {
        ਲਈ (uint32_t d = 0; d < S_v / 64; ++d) {
            ਅਚਲ HVX_ਸਦਿਸ਼ * t0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + (r0 * n_ਖਾਨਾ_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ਅਚਲ HVX_ਸਦਿਸ਼ * t1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + (r0 * n_ਖਾਨਾ_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            ਲਈ (uint32_t r = 0; r < 16; ++r) {
                HVX_ਸਦਿਸ਼_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_unpack_Sx_S_tiles_ਮੰਜ਼ਿਲ_f32(
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_f32,
    ਅਚਲ __fp16 * ਸੀਮਿਤ tiles,
    uint32_t S_v
) {
    ਅਚਲ uint32_t n_tiles = S_v / 32;
    ਲਈ (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        ਲਈ (uint32_t d = 0; d < S_v / 64; ++d) {
            ਅਚਲ HVX_ਸਦਿਸ਼ * t0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ਅਚਲ HVX_ਸਦਿਸ਼ * t1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            ਲਈ (uint32_t r = 0; r < 16; ++r) {
                HVX_ਸਦਿਸ਼_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_ਸਦਿਸ਼_Pair p0 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(Q6_V_lo_W(vp01));
                HVX_ਸਦਿਸ਼_Pair p1 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(Q6_V_hi_W(vp01));

                ਦਸ਼ਮਲਵ * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                ਦਸ਼ਮਲਵ * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_f32_ਮੰਜ਼ਿਲ_hmx_ਪੰਕਤੀ_tiles_and_f16(
    __fp16 * ਸੀਮਿਤ dst_tiles,
    __fp16 * ਸੀਮਿਤ dst_ਅਭਾਜ_tiles,
    __fp16 * ਸੀਮਿਤ dst_f16,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src,
    ਅਚਲ __fp16 * ਸੀਮਿਤ scale_ਪ੍ਰਤੀ_ਪੰਕਤੀ,
    uint32_t n_ਪੰਕਤੀਆਂ,
    uint32_t n_ਖਾਨੇ_ਗਿਣਤੀ
) {
    ਅਚਲ uint32_t n_ਖਾਨਾ_tiles = n_ਖਾਨੇ_ਗਿਣਤੀ / 32;
    ਅਚਲ uint32_t * scale_ਜੋੜੇ = (ਅਚਲ uint32_t *) scale_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        ਅਚਲ ਦਸ਼ਮਲਵ * p0 = src + (r + 0) * n_ਖਾਨੇ_ਗਿਣਤੀ;
        ਅਚਲ ਦਸ਼ਮਲਵ * p1 = src + (r + 1) * n_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ v_scale;
        ਜੇ (dst_ਅਭਾਜ_tiles) {
            uint32_t scale_pair = scale_ਜੋੜੇ ? scale_ਜੋੜੇ[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        ਲਈ (uint32_t c = 0; c < n_ਖਾਨਾ_tiles; c += 2) {
            HVX_ਸਦਿਸ਼ v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_ਸਦਿਸ਼ v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_ਸਦਿਸ਼ v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_ਸਦਿਸ਼ v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_ਸਦਿਸ਼ vh0 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0_0, v1_0);
            HVX_ਸਦਿਸ਼ vh1 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_ਖਾਨਾ_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_ਖਾਨਾ_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_ਸਦਿਸ਼ *) tile0)[r1] = vh0;
            ((HVX_ਸਦਿਸ਼ *) tile1)[r1] = vh1;

            ਜੇ (dst_ਅਭਾਜ_tiles) {
                HVX_ਸਦਿਸ਼ vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_ਸਦਿਸ਼ vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_ਅਭਾਜ_tiles + (r0 * n_ਖਾਨਾ_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_ਅਭਾਜ_tiles + (r0 * n_ਖਾਨਾ_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_ਸਦਿਸ਼ *) tile0_s)[r1] = vh0_s;
                ((HVX_ਸਦਿਸ਼ *) tile1_s)[r1] = vh1_s;
            }

            ਜੇ (dst_f16) {
                HVX_ਸਦਿਸ਼_Pair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_ਖਾਨੇ_ਗਿਣਤੀ + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_ਖਾਨੇ_ਗਿਣਤੀ + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_transpose_32x32_ਸ਼ਬਦ(HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ m, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tmp) {
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 4) {
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 8) {
        ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
            HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 16) {
        ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
            HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_pack_d_t_ਪੰਕਤੀ_tiles(
    __fp16 * ਸੀਮਿਤ dst_tiles,
    ਅਚਲ __fp16 * ਸੀਮਿਤ src_d,
    uint32_t S_v,
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ m,
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tmp
) {
    ਲਈ (uint32_t ਖਾਨਾ_half = 0; ਖਾਨਾ_half < S_v / 64; ++ਖਾਨਾ_half) {
        uint32_t r0_ਅਧਾਰ = ਖਾਨਾ_half * 2;
        ਲਈ (uint32_t c0 = 0; c0 < 2; ++c0) {
            ਲਈ (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + ਖਾਨਾ_half * 64);
            }

            hvx_transpose_32x32_ਸ਼ਬਦ(m, tmp);

            uint32_t tile0_idx = (r0_ਅਧਾਰ + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_ਅਧਾਰ + 1) * 2 + c0;
            HVX_ਸਦਿਸ਼ * t0 = (HVX_ਸਦਿਸ਼ *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_ਸਦਿਸ਼ * t1 = (HVX_ਸਦਿਸ਼ *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            ਲਈ (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ gdn_build_inv_l_ਖੰਡਾਂ(
    __fp16 * ਸੀਮਿਤ inv_ਪੰਕਤੀ_tiles,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ ਪੰਕਤੀਆਂ_kk,
    ਅਚਲ __fp16 * ਸੀਮਿਤ decay_m,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ beta,
    __fp16 * ਸੀਮਿਤ l10_tile,
    __fp16 * ਸੀਮਿਤ neg_a11_tile
) {
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one_f16 = hvx_vec_splat_f16(1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ p_l00 = (HVX_ਸਦਿਸ਼ *) l00;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ p_l11 = (HVX_ਸਦਿਸ਼ *) l11;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ p_l10_tile = (HVX_ਸਦਿਸ਼ *) l10_tile;

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tile00 = (HVX_ਸਦਿਸ਼ *) (inv_ਪੰਕਤੀ_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tile01 = (HVX_ਸਦਿਸ਼ *) (inv_ਪੰਕਤੀ_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tile11 = (HVX_ਸਦਿਸ਼ *) (inv_ਪੰਕਤੀ_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ p_neg_a11 = (HVX_ਸਦਿਸ਼ *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_ਸਦਿਸ਼ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_ਸਦਿਸ਼ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_ਸਦਿਸ਼ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_ਸਦਿਸ਼ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_ਸਦਿਸ਼ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ਪੰਕਤੀਆਂ_kk[t0], v_d0), v_b0);
        HVX_ਸਦਿਸ਼ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ਪੰਕਤੀਆਂ_kk[t1], v_d1), v_b1);

        p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
    }

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 32 + 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_ਸਦਿਸ਼ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_ਸਦਿਸ਼ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_ਸਦਿਸ਼ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_ਸਦਿਸ਼ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_ਸਦਿਸ਼ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ਪੰਕਤੀਆਂ_kk[t0], v_d0), v_b0);
        HVX_ਸਦਿਸ਼ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ਪੰਕਤੀਆਂ_kk[t1], v_d1), v_b1);

        HVX_ਸਦਿਸ਼_Pair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
        p_l10_tile[r] = Q6_V_lo_W(vp_l10);
        p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
    }

    HVX_ਸਦਿਸ਼ a_ਪੰਕਤੀਆਂ[32];
    ਲਈ (uint32_t t = 0; t < 32; ++t) {
        HVX_ਸਦਿਸ਼ v_inv = Q6_V_vzero();
        ਲਈ (uint32_t k = 0; k < t; ++k) {
            HVX_ਸਦਿਸ਼ v_lk = Q6_Vh_vsplat_R(l00[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_ਪੰਕਤੀਆਂ[k]));
        }
        HVX_ਸਦਿਸ਼_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_ਪੰਕਤੀਆਂ[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(a_ਪੰਕਤੀਆਂ[2 * r + 1], a_ਪੰਕਤੀਆਂ[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    ਲਈ (uint32_t t = 0; t < 32; ++t) {
        HVX_ਸਦਿਸ਼ v_inv = Q6_V_vzero();
        ਲਈ (uint32_t k = 0; k < t; ++k) {
            HVX_ਸਦਿਸ਼ v_lk = Q6_Vh_vsplat_R(l11[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_ਪੰਕਤੀਆਂ[k]));
        }
        HVX_ਸਦਿਸ਼_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_ਪੰਕਤੀਆਂ[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    ਲਈ (uint32_t r = 0; r < 16; ++r) {
        HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(a_ਪੰਕਤੀਆਂ[2 * r + 1], a_ਪੰਕਤੀਆਂ[2 * r + 0], -2);
        tile11[r] = Q6_V_lo_W(vp);

        HVX_ਸਦਿਸ਼ n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_ਪੰਕਤੀਆਂ[2 * r + 0]);
        HVX_ਸਦਿਸ਼ n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_ਪੰਕਤੀਆਂ[2 * r + 1]);
        HVX_ਸਦਿਸ਼_Pair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
        p_neg_a11[r] = Q6_V_lo_W(vp_neg);
    }
}


ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_dma_ਧੱਕੋ_chunk_inputs(
    dma_ਕਤਾਰ * dma_q,
    ਦਸ਼ਮਲਵ * vtcm_q,
    ਦਸ਼ਮਲਵ * vtcm_k,
    ਦਸ਼ਮਲਵ * vtcm_v,
    ਅਚਲ ਬਣਤਰ htp_tensor * q,
    ਅਚਲ ਬਣਤਰ htp_tensor * k,
    ਅਚਲ ਬਣਤਰ htp_tensor * v,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_ਮਾਪ,
    uint32_t S_v
) {
    ਅਚਲ dma_addr_t q_dma = q->ਅੰਕੜੇ + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
    ਅਚਲ dma_addr_t k_dma = k->ਅੰਕੜੇ + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
    ਅਚਲ dma_addr_t v_dma = v->ਅੰਕੜੇ + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_q, q_dma), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), q->nb[2], S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), chunk_ਮਾਪ);
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_k, k_dma), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k->nb[2], S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), chunk_ਮਾਪ);
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_v, v_dma), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), v->nb[2], S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), chunk_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_dma_ਧੱਕੋ_chunk_gb(
    dma_ਕਤਾਰ * dma_q,
    ਦਸ਼ਮਲਵ * vtcm_g_raw,
    ਦਸ਼ਮਲਵ * vtcm_b_raw,
    ਅਚਲ ਬਣਤਰ htp_tensor * g,
    ਅਚਲ ਬਣਤਰ htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_ਮਾਪ,
    uint32_t n_batch
) {
    ਅਚਲ dma_addr_t g_dma    = g->ਅੰਕੜੇ + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    ਅਚਲ dma_addr_t beta_dma = beta->ਅੰਕੜੇ + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਬਾਈਟਾਂ  = n_batch * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_g_raw, g_dma), ਪੰਕਤੀ_ਬਾਈਟਾਂ, g->nb[2], ਪੰਕਤੀ_ਬਾਈਟਾਂ, chunk_ਮਾਪ);
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_b_raw, beta_dma), ਪੰਕਤੀ_ਬਾਈਟਾਂ, beta->nb[2], ਪੰਕਤੀ_ਬਾਈਟਾਂ, chunk_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_pack_s_ਖਾਨਾ_tiles(
    __fp16 * ਸੀਮਿਤ vtcm_s_ਖਾਨਾ_tiles,
    __fp16 * ਸੀਮਿਤ vtcm_s_f16,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ vtcm_s_state,
    uint32_t S_v
) {
    ਲਈ (uint32_t j = 0; j < S_v; ++j) {
        ਲਈ (uint32_t i = 0; i < S_v; i += 64) {
            HVX_ਸਦਿਸ਼ v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_ਸਦਿਸ਼ v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v0, v1);
        }
    }
    hmx_interleave_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_tiles(vtcm_s_ਖਾਨਾ_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);
}

ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs {
    ਦਸ਼ਮਲਵ *  s_state;
    __fp16 * s_f16;
    __fp16 * s_ਖਾਨਾ_tiles;
    ਦਸ਼ਮਲਵ *  s_update_f32;
    __fp16 * s_update_tiles;

    ਦਸ਼ਮਲਵ * q_f32[2];
    ਦਸ਼ਮਲਵ * k_f32[2];
    ਦਸ਼ਮਲਵ * v_f32[2];
    ਦਸ਼ਮਲਵ * g_f32[2];
    ਦਸ਼ਮਲਵ * b_f32[2];
    ਦਸ਼ਮਲਵ * o_f32[2];

    ਦਸ਼ਮਲਵ * v_inter_f32;
    ਦਸ਼ਮਲਵ * o_inter_f32;
    ਦਸ਼ਮਲਵ * o_intra_f32;

    __fp16 * k_f16;
    __fp16 * v_ਅਭਾਜ_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_ਪੰਕਤੀ_tiles;
    __fp16 * q_ਅਭਾਜ_ਪੰਕਤੀ_tiles;
    __fp16 * k_ਪੰਕਤੀ_tiles;
    __fp16 * k_ਖਾਨਾ_tiles;
    __fp16 * k_ਅਭਾਜ_ਪੰਕਤੀ_tiles;
    __fp16 * k_ਖਾਨਾ_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_ਪੰਕਤੀ_tiles;
    __fp16 * a_ਪੰਕਤੀ_tiles;
    __fp16 * v_ਅਭਾਜ_ਖਾਨਾ_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_ਖਾਨਾ_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_ਪੰਕਤੀ_tiles;

    __fp16 * gamma;
    ਦਸ਼ਮਲਵ *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_ਸਦਿਸ਼ * ਪੰਕਤੀਆਂ_kk;
    HVX_ਸਦਿਸ਼ * ਪੰਕਤੀਆਂ_qk;
    HVX_ਸਦਿਸ਼ * ਪੰਕਤੀਆਂ_inv;
    HVX_ਸਦਿਸ਼ * ਪੰਕਤੀਆਂ_a;

    HVX_ਸਦਿਸ਼ * vtcm_m;
    HVX_ਸਦਿਸ਼ * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_ਵਿੱਚ_dma;
    dma_addr_t state_ਬਾਹਰਲਾ_dma;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gdn_init_ਸਿਰ_ptrs(
    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ,
    ਅਚਲ ਬਣਤਰ htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_ਅਧਾਰ,
    uint32_t h,
    uint32_t ਅਧਾਰ_iv1,
    uint32_t iv3,
    ਅਚਲ ਬਣਤਰ htp_tensor * q,
    ਅਚਲ ਬਣਤਰ htp_tensor * k,
    ਅਚਲ ਬਣਤਰ htp_tensor * v,
    ਅਚਲ ਬਣਤਰ htp_tensor * state,
    ਅਚਲ ਬਣਤਰ htp_tensor * dst,
    ਅਚਲ ਬਣਤਰ htp_tensor * dst_cache,
    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_ਟੋਕਨ,
    uint32_t chunk_ਮਾਪ
) {
    ਅਚਲ size_t dma_scalar_sz = hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    ਅਚਲ size_t decay_sz      = 64 * 64 * ਆਕਾਰ(__fp16);
    ਅਚਲ size_t ਪੰਕਤੀ_vecs_sz   = 64 * 128;

    ਸਿਰ->s_state        = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_s_state + h * L->state_f32_ਬਾਈਟਾਂ);
    ਸਿਰ->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_s_f16 + h * L->state_f16_ਬਾਈਟਾਂ);
    ਸਿਰ->s_ਖਾਨਾ_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_s_ਖਾਨਾ_tiles + h * L->state_tiles_ਬਾਈਟਾਂ);
    ਸਿਰ->s_update_f32   = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_s_update_f32 + h * L->state_f32_ਬਾਈਟਾਂ);
    ਸਿਰ->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_s_update_tiles + h * L->state_tiles_ਬਾਈਟਾਂ);

    ਸਿਰ->q_f32[0] = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_q_f32[0] + h * L->dma_chunk_ਬਾਈਟਾਂ);
    ਸਿਰ->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_q_f32[1] + h * L->dma_chunk_ਬਾਈਟਾਂ) : ਸਿਰ->q_f32[0];
    ਸਿਰ->k_f32[0] = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_k_f32[0] + h * L->dma_chunk_ਬਾਈਟਾਂ);
    ਸਿਰ->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_k_f32[1] + h * L->dma_chunk_ਬਾਈਟਾਂ) : ਸਿਰ->k_f32[0];
    ਸਿਰ->v_f32[0] = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_v_f32[0] + h * L->dma_chunk_ਬਾਈਟਾਂ);
    ਸਿਰ->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_v_f32[1] + h * L->dma_chunk_ਬਾਈਟਾਂ) : ਸਿਰ->v_f32[0];
    ਸਿਰ->g_f32[0] = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_g_f32[0] + h * dma_scalar_sz);
    ਸਿਰ->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_g_f32[1] + h * dma_scalar_sz) : ਸਿਰ->g_f32[0];
    ਸਿਰ->b_f32[0] = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_b_f32[0] + h * dma_scalar_sz);
    ਸਿਰ->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_b_f32[1] + h * dma_scalar_sz) : ਸਿਰ->b_f32[0];
    ਸਿਰ->o_f32[0] = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_o_f32[0] + h * L->dma_chunk_ਬਾਈਟਾਂ);
    ਸਿਰ->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_o_f32[1] + h * L->dma_chunk_ਬਾਈਟਾਂ) : ਸਿਰ->o_f32[0];

    ਸਿਰ->v_inter_f32 = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_v_inter_f32 + h * L->dma_chunk_ਬਾਈਟਾਂ);
    ਸਿਰ->o_inter_f32 = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_o_inter_f32 + h * L->dma_chunk_ਬਾਈਟਾਂ);
    ਸਿਰ->o_intra_f32 = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_o_intra_f32 + h * L->dma_chunk_ਬਾਈਟਾਂ);

    ਸਿਰ->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_k_f16 + h * L->act_f16_ਬਾਈਟਾਂ);
    ਸਿਰ->v_ਅਭਾਜ_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_v_ਅਭਾਜ_f16 + h * L->act_f16_ਬਾਈਟਾਂ);
    ਸਿਰ->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_delta_f16 + h * L->act_f16_ਬਾਈਟਾਂ);
    ਸਿਰ->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_d_f16 + h * L->act_f16_ਬਾਈਟਾਂ);

    ਸਿਰ->q_ਪੰਕਤੀ_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_q_ਪੰਕਤੀ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->q_ਅਭਾਜ_ਪੰਕਤੀ_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_q_ਅਭਾਜ_ਪੰਕਤੀ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->k_ਪੰਕਤੀ_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_k_ਪੰਕਤੀ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->k_ਖਾਨਾ_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_k_ਖਾਨਾ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->k_ਅਭਾਜ_ਪੰਕਤੀ_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_k_ਅਭਾਜ_ਪੰਕਤੀ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->k_ਖਾਨਾ_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_k_ਖਾਨਾ_tiles_64x128 + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_kk_tiles + h * L->tile_64x64_ਬਾਈਟਾਂ);
    ਸਿਰ->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_qk_tiles + h * L->tile_64x64_ਬਾਈਟਾਂ);
    ਸਿਰ->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_v_inter_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_o_inter_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->inv_ਪੰਕਤੀ_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_inv_ਪੰਕਤੀ_tiles + h * L->tile_64x64_ਬਾਈਟਾਂ);
    ਸਿਰ->a_ਪੰਕਤੀ_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_a_ਪੰਕਤੀ_tiles + h * L->tile_64x64_ਬਾਈਟਾਂ);
    ਸਿਰ->v_ਅਭਾਜ_ਖਾਨਾ_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_v_ਅਭਾਜ_ਖਾਨਾ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_delta_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->delta_ਖਾਨਾ_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_delta_ਖਾਨਾ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_o_intra_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);
    ਸਿਰ->d_ਪੰਕਤੀ_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_d_ਪੰਕਤੀ_tiles + h * L->tile_64x_Sv_ਬਾਈਟਾਂ);

    ਸਿਰ->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_gamma + h * dma_scalar_sz);
    ਸਿਰ->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_gamma + h * dma_scalar_sz + 128);
    ਸਿਰ->lambda_init     = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L->off_lambda_init + h * dma_scalar_sz);
    ਸਿਰ->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_decay_m + h * decay_sz);
    ਸਿਰ->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_ਅਧਾਰ, L->off_decay_a + h * decay_sz);

    ਸਿਰ->ਪੰਕਤੀਆਂ_kk  = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, vtcm_ਅਧਾਰ, L->off_ਪੰਕਤੀਆਂ_kk + h * ਪੰਕਤੀ_vecs_sz);
    ਸਿਰ->ਪੰਕਤੀਆਂ_qk  = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, vtcm_ਅਧਾਰ, L->off_ਪੰਕਤੀਆਂ_qk + h * ਪੰਕਤੀ_vecs_sz);
    ਸਿਰ->ਪੰਕਤੀਆਂ_inv = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, vtcm_ਅਧਾਰ, L->off_ਪੰਕਤੀਆਂ_inv + h * ਪੰਕਤੀ_vecs_sz);
    ਸਿਰ->ਪੰਕਤੀਆਂ_a   = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, vtcm_ਅਧਾਰ, L->off_ਪੰਕਤੀਆਂ_a + h * ਪੰਕਤੀ_vecs_sz);

    ਸਿਰ->vtcm_m   = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, vtcm_ਅਧਾਰ, L->off_thread_scratch + h * (64 * 128));
    ਸਿਰ->vtcm_tmp = ਸਿਰ->vtcm_m + 32;

    ਸਿਰ->iv1 = ਅਧਾਰ_iv1 + h;
    ਸਿਰ->iv3 = iv3;
    ਸਿਰ->iq1 = fastmodulo(ਸਿਰ->iv1, q->ne[1], &kparams->div_q1);
    ਸਿਰ->ik1 = fastmodulo(ਸਿਰ->iv1, k->ne[1], &kparams->div_k1);
    ਸਿਰ->iq3 = fastdiv(ਸਿਰ->iv3, &kparams->div_rq3);
    ਸਿਰ->ik3 = fastdiv(ਸਿਰ->iv3, &kparams->div_rk3);

    ਸਿਰ->state_ਵਿੱਚ_dma = state->ਅੰਕੜੇ +
        ((uint64_t) ਸਿਰ->iv3 * kparams->state_seq_stride + (uint64_t) ਸਿਰ->iv1 * S_v * S_v) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਸਿਰ->state_ਬਾਹਰਲਾ_dma = dst_cache ?
        (dst_cache->ਅੰਕੜੇ + ((uint64_t) ਸਿਰ->iv3 * H + ਸਿਰ->iv1) * S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) :
        (dst->ਅੰਕੜੇ + ((uint64_t) S_v * H * n_ਟੋਕਨ * kparams->n_seqs + (uint64_t) (ਸਿਰ->iv3 * H + ਸਿਰ->iv1) * S_v * S_v) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

ਬਣਤਰ htp_gdn_batch_context {
    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * heads;
    ਅਚਲ ਦਸ਼ਮਲਵ *              vtcm_g_raw;
    ਅਚਲ ਦਸ਼ਮਲਵ *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    ਦਸ਼ਮਲਵ                      scale;
    ਬਣਤਰ htp_ops_context *   octx;
    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams;
};

ਸਥਿਰ ਖਾਲੀ gdn_hvx_init_state_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    gdn_pack_s_ਖਾਨਾ_tiles(ਸਿਰ->s_ਖਾਨਾ_tiles, ਸਿਰ->s_f16, ਸਿਰ->s_state, bctx->S_v);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ __attribute__((ਅਣਵਰਤਿਆ_2)) HVX_ਸਦਿਸ਼ hvx_clamp_neg20_0(HVX_ਸਦਿਸ਼ v, HVX_ਸਦਿਸ਼ v_zero, HVX_ਸਦਿਸ਼ v_neg20) {
    HVX_ਸਦਿਸ਼_Pred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_ਸਦਿਸ਼_Pred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    ਵਾਪਸ Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_ਅਗੇਤਰ_ਜਾਂਚ_f32(HVX_ਸਦਿਸ਼ v, HVX_ਸਦਿਸ਼ ਹਾਸਲ_ਵਿੱਚ) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

    v = hvx_vec_ਜੋੜੋ_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_ਜੋੜੋ_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_ਜੋੜੋ_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_ਜੋੜੋ_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_ਜੋੜੋ_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_ਜੋੜੋ_f32_f32(v, ਹਾਸਲ_ਵਿੱਚ);

    ਵਾਪਸ v;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_splat_ਆਖਰੀ_f32(HVX_ਸਦਿਸ਼ v) {
    ਵਾਪਸ hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase1a_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t curr_buf = bctx->curr_buf;
    ਅਚਲ uint32_t S_v = bctx->S_v;
    ਅਚਲ uint32_t n_batch = bctx->n_batch;

    ਜੇ (n_batch == 1) {
        hvx_vmem(ਸਿਰ->g_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(ਸਿਰ->g_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(ਸਿਰ->b_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(ਸਿਰ->b_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } ਨਹੀਂਤਾਂ {
        int32_t offsets[32] __attribute__((aligned(128)));
        ਲਈ (ਪੂਰਨ k = 0; k < 32; ++k) {
            offsets[k] = k * n_batch * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        }
        HVX_ਸਦਿਸ਼ vv = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) offsets;
        ਅਚਲ size_t rt_g = (size_t) ((ਅਚਲ uint8_t *) bctx->vtcm_g_raw + i * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ size_t rt_b = (size_t) ((ਅਚਲ uint8_t *) bctx->vtcm_b_raw + i * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ size_t mu   = 64 * n_batch * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (ਸਿਰ->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (ਸਿਰ->g_f32[curr_buf] + 32), rt_g + 32 * n_batch * ਆਕਾਰ(ਦਸ਼ਮਲਵ), mu, vv);
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (ਸਿਰ->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (ਸਿਰ->b_f32[curr_buf] + 32), rt_b + 32 * n_batch * ਆਕਾਰ(ਦਸ਼ਮਲਵ), mu, vv);
    }

    ਅਚਲ uint32_t t_chunk = bctx->c * 64;
    ਅਚਲ uint32_t valid_ਟੋਕਨ = hex_smin(64, bctx->kparams->n_ਟੋਕਨ - t_chunk);
    ਜੇ (valid_ਟੋਕਨ < 64) {
        ਲਈ (uint32_t t = valid_ਟੋਕਨ; t < 64; ++t) {
            ਸਿਰ->g_f32[curr_buf][t] = 0.0f;
            ਸਿਰ->b_f32[curr_buf][t] = 0.0f;
        }
        ਅਚਲ HVX_ਸਦਿਸ਼ vzero = Q6_V_vzero();
        ਲਈ (uint32_t t = valid_ਟੋਕਨ; t < 64; ++t) {
            ਲਈ (uint32_t j = 0; j < S_v; j += 32) {
                hvx_vmem(ਸਿਰ->q_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(ਸਿਰ->k_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(ਸਿਰ->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    ਅਚਲ HVX_ਸਦਿਸ਼ v_g0 = hvx_vmem(ਸਿਰ->g_f32[curr_buf] + 0);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_g1 = hvx_vmem(ਸਿਰ->g_f32[curr_buf] + 32);

    HVX_ਸਦਿਸ਼ v_gamma0 = hvx_ਅਗੇਤਰ_ਜਾਂਚ_f32(v_g0, Q6_V_vzero());
    HVX_ਸਦਿਸ਼ v_ਹਾਸਲ  = hvx_splat_ਆਖਰੀ_f32(v_gamma0);
    HVX_ਸਦਿਸ਼ v_gamma1 = hvx_ਅਗੇਤਰ_ਜਾਂਚ_f32(v_g1, v_ਹਾਸਲ);

    ਅਚਲ HVX_ਸਦਿਸ਼ v_zero  = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ v_neg20 = hvx_vec_splat_f32(-20.0f);

    hvx_vmem(ਸਿਰ->gamma) = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v_gamma0, v_gamma1);

    HVX_ਸਦਿਸ਼ v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_ਸਦਿਸ਼ v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    hvx_vmem(ਸਿਰ->lambda_init + 0)  = v_l0;
    hvx_vmem(ਸਿਰ->lambda_init + 32) = v_l1;
    hvx_vmem(ਸਿਰ->lambda_init_f16)  = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v_l0, v_l1);

    gdn_f32_ਮੰਜ਼ਿਲ_hmx_ਪੰਕਤੀ_tiles_and_f16(ਸਿਰ->k_ਪੰਕਤੀ_tiles, ਸਿਰ->k_ਅਭਾਜ_ਪੰਕਤੀ_tiles, ਸਿਰ->k_f16,
                                     ਸਿਰ->k_f32[curr_buf], ਸਿਰ->lambda_init_f16, 64, S_v);
    hmx_interleave_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_tiles(ਸਿਰ->k_ਖਾਨਾ_tiles, ਸਿਰ->k_f16, 64, S_v, S_v, 0, 64);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase1b_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t curr_buf = bctx->curr_buf;
    ਅਚਲ uint32_t S_v = bctx->S_v;

    gdn_f32_ਮੰਜ਼ਿਲ_hmx_ਪੰਕਤੀ_tiles_and_f16(ਸਿਰ->q_ਪੰਕਤੀ_tiles, ਸਿਰ->q_ਅਭਾਜ_ਪੰਕਤੀ_tiles, ਨਿਰਾ,
                                     ਸਿਰ->q_f32[curr_buf], ਸਿਰ->lambda_init_f16, 64, S_v);

    hmx_interleave_ਖਾਨੇ_ਗਿਣਤੀ_ਮੰਜ਼ਿਲ_tiles(ਸਿਰ->k_ਖਾਨਾ_tiles_64x128, ਸਿਰ->k_f16, 64, S_v, S_v, 2, 0, 64);

    ਅਚਲ uint16_t * gamma_u16 = (ਅਚਲ uint16_t *) ਸਿਰ->gamma;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_gamma   = hvx_vmem(ਸਿਰ->gamma);

    ਅਚਲ HVX_ਸਦਿਸ਼ v_zero_f16  = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(ਸਿਰ->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(ਸਿਰ->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    ਲਈ (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_ਸਦਿਸ਼ v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_ਸਦਿਸ਼ v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_ਸਦਿਸ਼ diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_ਸਦਿਸ਼ diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_ਸਦਿਸ਼_Pred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_ਸਦਿਸ਼_Pred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_ਸਦਿਸ਼ diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_ਸਦਿਸ਼ diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_ਸਦਿਸ਼ v_exp0      = hvx_vec_exp2_f16(diff_log2e0);
        HVX_ਸਦਿਸ਼ v_exp1      = hvx_vec_exp2_f16(diff_log2e1);

        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_ਸਦਿਸ਼ v_m0         = Q6_V_vand_QV(ਨਕਾਬ_lt0, v_exp0);
        HVX_ਸਦਿਸ਼ v_m1         = Q6_V_vand_QV(ਨਕਾਬ_lt1, v_exp1);

        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_diag0 = Q6_Q_and_QQn(ਨਕਾਬ_le0, ਨਕਾਬ_lt0);
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_diag1 = Q6_Q_and_QQn(ਨਕਾਬ_le1, ਨਕਾਬ_lt1);

        HVX_ਸਦਿਸ਼ v_a0         = Q6_V_vmux_QVV(ਨਕਾਬ_diag0, v_one_f16, v_m0);
        HVX_ਸਦਿਸ਼ v_a1         = Q6_V_vmux_QVV(ਨਕਾਬ_diag1, v_one_f16, v_m1);

        hvx_vmem(ਸਿਰ->decay_m + t0 * 64) = v_m0;
        hvx_vmem(ਸਿਰ->decay_a + t0 * 64) = v_a0;
        hvx_vmem(ਸਿਰ->decay_m + t1 * 64) = v_m1;
        hvx_vmem(ਸਿਰ->decay_a + t1 * 64) = v_a1;
    }

    {
        HVX_ਸਦਿਸ਼ v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_ਸਦਿਸ਼ diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_ਸਦਿਸ਼_Pred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_ਸਦਿਸ਼ diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_ਸਦਿਸ਼ v_exp         = hvx_vec_exp2_f16(diff_log2e);

        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_ਸਦਿਸ਼ v_m           = Q6_V_vand_QV(ਨਕਾਬ_lt_t, v_exp);

        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_diag = Q6_Q_and_QQn(ਨਕਾਬ_le_t, ਨਕਾਬ_lt_t);
        HVX_ਸਦਿਸ਼ v_a           = Q6_V_vmux_QVV(ਨਕਾਬ_diag, v_one_f16, v_m);

        hvx_vmem(ਸਿਰ->decay_m + 63 * 64) = v_m;
        hvx_vmem(ਸਿਰ->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase2_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t curr_buf = bctx->curr_buf;

    gdn_unpack_64x64_tiles_ਮੰਜ਼ਿਲ_vectors(ਸਿਰ->ਪੰਕਤੀਆਂ_kk, ਸਿਰ->kk_tiles);

    gdn_build_inv_l_ਖੰਡਾਂ(
        ਸਿਰ->inv_ਪੰਕਤੀ_tiles,
        ਸਿਰ->ਪੰਕਤੀਆਂ_kk,
        ਸਿਰ->decay_m,
        ਸਿਰ->b_f32[curr_buf],
        (__fp16 *) ਸਿਰ->vtcm_m,
        (__fp16 *) ਸਿਰ->vtcm_m + HMX_FP16_TILE_N_ELMS
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase3_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t curr_buf = bctx->curr_buf;
    ਅਚਲ uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_ਮੰਜ਼ਿਲ_f32(ਸਿਰ->v_inter_f32, ਸਿਰ->v_inter_tiles, S_v);

    HVX_ਸਦਿਸ਼_Alias local_b[2];
    local_b[0].v = hvx_vmem(ਸਿਰ->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(ਸਿਰ->b_f32[curr_buf] + 32);

    ਲਈ (uint32_t t = 0; t < 64; ++t) {
        HVX_ਸਦਿਸ਼ vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
        ਲਈ (uint32_t j = 0; j < S_v; j += 64) {
            HVX_ਸਦਿਸ਼ vv0 = hvx_vmem(ਸਿਰ->v_f32[curr_buf] + t * S_v + j + 0);
            HVX_ਸਦਿਸ਼ vv1 = (j + 32 < S_v) ? hvx_vmem(ਸਿਰ->v_f32[curr_buf] + t * S_v + j + 32) : Q6_V_vzero();
            HVX_ਸਦਿਸ਼ vi0 = hvx_vmem(ਸਿਰ->v_inter_f32 + t * S_v + j + 0);
            HVX_ਸਦਿਸ਼ vi1 = (j + 32 < S_v) ? hvx_vmem(ਸਿਰ->v_inter_f32 + t * S_v + j + 32) : Q6_V_vzero();

            HVX_ਸਦਿਸ਼ vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
            HVX_ਸਦਿਸ਼ vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

            hvx_vmem(ਸਿਰ->v_ਅਭਾਜ_f16 + t * S_v + j) = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(vp0, vp1);
        }
    }

    hmx_interleave_ਖਾਨੇ_ਗਿਣਤੀ_ਮੰਜ਼ਿਲ_tiles(ਸਿਰ->v_ਅਭਾਜ_ਖਾਨਾ_tiles, ਸਿਰ->v_ਅਭਾਜ_f16, 64, S_v, S_v, 2, 0, 64);

    gdn_unpack_64x64_tiles_ਮੰਜ਼ਿਲ_vectors(ਸਿਰ->ਪੰਕਤੀਆਂ_qk, ਸਿਰ->qk_tiles);
    ਲਈ (uint32_t t = 0; t < 64; ++t) {
        HVX_ਸਦਿਸ਼ v_decay_a = hvx_vmem(ਸਿਰ->decay_a + t * 64);
        ਸਿਰ->ਪੰਕਤੀਆਂ_a[t]      = hvx_vec_mul_f16_f16(ਸਿਰ->ਪੰਕਤੀਆਂ_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_ਮੰਜ਼ਿਲ_tiles(ਸਿਰ->a_ਪੰਕਤੀ_tiles, ਸਿਰ->ਪੰਕਤੀਆਂ_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase4_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_ਮੰਜ਼ਿਲ_f16(ਸਿਰ->delta_f16, ਸਿਰ->delta_tiles, S_v);
    hmx_interleave_ਖਾਨੇ_ਗਿਣਤੀ_ਮੰਜ਼ਿਲ_tiles(ਸਿਰ->delta_ਖਾਨਾ_tiles, ਸਿਰ->delta_f16, 64, S_v, S_v, 2, 0, 64);

    ਅਚਲ uint16_t * decay_ਆਖਰੀ = (ਅਚਲ uint16_t *) (ਸਿਰ->decay_a + 63 * 64);
    ਅਚਲ HVX_ਸਦਿਸ਼ vzero      = Q6_V_vzero();

    ਲਈ (uint32_t s = 0; s < 64; ++s) {
        HVX_ਸਦਿਸ਼ vs         = Q6_Vh_vsplat_R(decay_ਆਖਰੀ[s]);
        HVX_ਸਦਿਸ਼_Pred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
        ਲਈ (uint32_t j = 0; j < S_v; j += 64) {
            HVX_ਸਦਿਸ਼ vd   = hvx_vmem(ਸਿਰ->delta_f16 + s * S_v + j);
            HVX_ਸਦਿਸ਼ prod = hvx_vec_mul_f16_f16(vd, vs);
            hvx_vmem(ਸਿਰ->d_f16 + s * S_v + j) = Q6_V_vmux_QVV(p_zero, vzero, prod);
        }
    }

    gdn_pack_d_t_ਪੰਕਤੀ_tiles(ਸਿਰ->d_ਪੰਕਤੀ_tiles, ਸਿਰ->d_f16, S_v, ਸਿਰ->vtcm_m, ਸਿਰ->vtcm_tmp);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase5_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_ਬਾਹਰਲਾ, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t curr_buf = bctx->curr_buf;
    ਅਚਲ uint32_t S_v = bctx->S_v;
    ਅਚਲ ਦਸ਼ਮਲਵ scale = bctx->scale;

    gdn_unpack_64x_S_tiles_ਮੰਜ਼ਿਲ_f32(ਸਿਰ->o_inter_f32, ਸਿਰ->o_inter_tiles, S_v);
    gdn_unpack_64x_S_tiles_ਮੰਜ਼ਿਲ_f32(ਸਿਰ->o_intra_f32, ਸਿਰ->o_intra_tiles, S_v);

    HVX_ਸਦਿਸ਼ vscale = hvx_vec_splat_f32(scale);
    ਲਈ (uint32_t j = 0; j < 64 * S_v / 32; ++j) {
        HVX_ਸਦਿਸ਼ vi = hvx_vmem(ਸਿਰ->o_inter_f32 + j * 32);
        HVX_ਸਦਿਸ਼ va = hvx_vmem(ਸਿਰ->o_intra_f32 + j * 32);
        hvx_vmem(ਸਿਰ->o_f32[curr_buf] + j * 32) = hvx_vec_mul_f32_f32(hvx_vec_ਜੋੜੋ_f32_f32(vi, va), vscale);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_ਬਾਹਰਲਾ, info);
}

ਸਥਿਰ ਖਾਲੀ gdn_hvx_phase6_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    ਬਣਤਰ htp_gdn_batch_context * bctx = (ਬਣਤਰ htp_gdn_batch_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ਅਚਲ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs * ਸਿਰ = &bctx->heads[i];
    ਅਚਲ uint32_t S_v = bctx->S_v;
    ਅਚਲ uint32_t c = bctx->c;
    ਅਚਲ uint32_t n_chunks = bctx->kparams->n_chunks;

    gdn_unpack_Sx_S_tiles_ਮੰਜ਼ਿਲ_f32(ਸਿਰ->s_update_f32, ਸਿਰ->s_update_tiles, S_v);

    HVX_ਸਦਿਸ਼_Alias ਆਖਰੀ_lambda;
    ਆਖਰੀ_lambda.v = hvx_vmem(ਸਿਰ->lambda_init + 32);
    HVX_ਸਦਿਸ਼ v_l_final = hvx_vec_splat_f32(ਆਖਰੀ_lambda.fp32[31]);

    ਲਈ (uint32_t j = 0; j < S_v * S_v / 32; ++j) {
        HVX_ਸਦਿਸ਼ vs_old = hvx_vmem(ਸਿਰ->s_state + j * 32);
        HVX_ਸਦਿਸ਼ vsu    = hvx_vmem(ਸਿਰ->s_update_f32 + j * 32);
        hvx_vmem(ਸਿਰ->s_state + j * 32) = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
    }

    ਜੇ (c + 1 < n_chunks) {
        gdn_pack_s_ਖਾਨਾ_tiles(ਸਿਰ->s_ਖਾਨਾ_tiles, ਸਿਰ->s_f16, ਸਿਰ->s_state, S_v);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


ਸਥਿਰ ਪੂਰਨ gated_delta_net_f32_hmx_chunked(
    ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams,
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ,
    uint32_t nrows
) {
    ਅਚਲ ਬਣਤਰ htp_tensor * q         = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k         = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v         = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * g         = octx->src[3];
    ਅਚਲ ਬਣਤਰ htp_tensor * beta      = octx->src[4];
    ਅਚਲ ਬਣਤਰ htp_tensor * state     = octx->src[5];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst       = octx->dst;
    ਅਚਲ ਬਣਤਰ htp_tensor * dst_cache = octx->dsts[1];

    ਅਚਲ uint32_t S_v        = kparams->S_v;
    ਅਚਲ uint32_t H          = kparams->H;
    ਅਚਲ uint32_t n_ਟੋਕਨ   = kparams->n_ਟੋਕਨ;
    ਅਚਲ ਦਸ਼ਮਲਵ    scale      = kparams->scale;
    ਅਚਲ uint32_t chunk_ਮਾਪ = kparams->chunk_ਮਾਪ;
    ਅਚਲ uint32_t n_chunks   = kparams->n_chunks;
    ਅਚਲ uint32_t n_sv_tiles = S_v / 32;

    ਬਣਤਰ htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_ਮਾਪ, kparams->n_heads_batch, kparams->n_ਧਾਗੇ, kparams->pipeline != 0);

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ vtcm_ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;

    ਦਸ਼ਮਲਵ * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L.off_g_raw[0])
    };
    ਦਸ਼ਮਲਵ * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, vtcm_ਅਧਾਰ, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_ਅਧਾਰ, L.off_scales_1);
    hmx_init_ਕਾਲਮ_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_ਕਤਾਰ_t  hmx_q = octx->ctx->hmx_ਕਤਾਰ;
    dma_ਕਤਾਰ *  dma_q = octx->ctx->dma[0];
    work_ਕਤਾਰ_t wp    = octx->ctx->work_ਕਤਾਰ;

    ਬਣਤਰ htp_gdn_ਸਿਰ_ptrs heads[8];
    ਬਣਤਰ htp_gdn_hmx_gemm_task gemm_tasks[8][9];

    uint32_t n_batch = 1;
    ਲਈ (uint32_t r = ਪੰਕਤੀ_ਸ਼ੁਰੂ; r < ਪੰਕਤੀ_ਸ਼ੁਰੂ + nrows; r += n_batch) {
        ਅਚਲ uint32_t ਸਿਰ_ਵਿੱਚ_seq         = fastmodulo(r, H, &kparams->div_H);
        ਅਚਲ uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        ਅਚਲ uint32_t heads_ਖੱਬਾ_ਵਿੱਚ_seq   = H - ਸਿਰ_ਵਿੱਚ_seq;
        ਅਚਲ uint32_t heads_ਖੱਬਾ_ਵਿੱਚ_range = (ਪੰਕਤੀ_ਸ਼ੁਰੂ + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_ਖੱਬਾ_ਵਿੱਚ_seq, heads_ਖੱਬਾ_ਵਿੱਚ_range));

        ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_ਸਿਰ_ptrs(&heads[h], &L, vtcm_ਅਧਾਰ, h, ਸਿਰ_ਵਿੱਚ_seq, iv3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_ਟੋਕਨ, chunk_ਮਾਪ);
        }

        ਬਣਤਰ htp_gdn_batch_context bctx;
        bctx.heads      = heads;
        bctx.vtcm_g_raw = ਨਿਰਾ;
        bctx.vtcm_b_raw = ਨਿਰਾ;
        bctx.curr_buf   = 0;
        bctx.c          = 0;
        bctx.n_batch    = n_batch;
        bctx.S_v        = S_v;
        bctx.scale      = scale;
        bctx.octx       = octx;
        bctx.kparams    = kparams;

        ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(heads[h].s_state, heads[h].state_ਵਿੱਚ_dma),
                           S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);
        }
        ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }

        ਜੇ (n_chunks > 0) {
            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_init_state_ਕਾਮਾ, &bctx, n_batch);

            ਅਚਲ uint32_t chunk0_ਟੋਕਨ = hex_smin(chunk_ਮਾਪ, n_ਟੋਕਨ);
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                gdn_dma_ਧੱਕੋ_chunk_inputs(dma_q, heads[h].q_f32[0], heads[h].k_f32[0], heads[h].v_f32[0],
                                          q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                          heads[h].iv3, heads[h].iv1, 0, chunk0_ਟੋਕਨ, S_v);
            }
            gdn_dma_ਧੱਕੋ_chunk_gb(dma_q, vtcm_g_raw[0], vtcm_b_raw[0], g, beta, iv3, ਸਿਰ_ਵਿੱਚ_seq, 0, chunk0_ਟੋਕਨ, n_batch);
        }

        ਲਈ (uint32_t c = 0; c < n_chunks; ++c) {
            ਅਚਲ uint32_t curr_buf = c & 1;
            ਅਚਲ uint32_t ਅਗਲਾ_buf = (c + 1) & 1;
            ਅਚਲ uint32_t t_chunk  = c * chunk_ਮਾਪ;

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
                dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
                dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
            }
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

            ਜੇ (c + 1 < n_chunks) {
                ਅਚਲ uint32_t ਅਗਲਾ_t_chunk = (c + 1) * chunk_ਮਾਪ;
                ਅਚਲ uint32_t ਅਗਲਾ_ਟੋਕਨ_2  = hex_smin(chunk_ਮਾਪ, n_ਟੋਕਨ - ਅਗਲਾ_t_chunk);
                ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_ਧੱਕੋ_chunk_inputs(dma_q, heads[h].q_f32[ਅਗਲਾ_buf], heads[h].k_f32[ਅਗਲਾ_buf], heads[h].v_f32[ਅਗਲਾ_buf],
                                              q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                              heads[h].iv3, heads[h].iv1, ਅਗਲਾ_t_chunk, ਅਗਲਾ_ਟੋਕਨ_2, S_v);
                }
                gdn_dma_ਧੱਕੋ_chunk_gb(dma_q, vtcm_g_raw[ਅਗਲਾ_buf], vtcm_b_raw[ਅਗਲਾ_buf],
                                      g, beta, iv3, ਸਿਰ_ਵਿੱਚ_seq, ਅਗਲਾ_t_chunk, ਅਗਲਾ_ਟੋਕਨ_2, n_batch);
            }

            ਜੇ (c > 0) {
                ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
                }
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase1a_ਕਾਮਾ, &bctx, n_batch);

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][0], heads[h].k_ਪੰਕਤੀ_tiles, heads[h].k_ਖਾਨਾ_tiles, heads[h].kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][2], heads[h].k_ਅਭਾਜ_ਪੰਕਤੀ_tiles, heads[h].s_ਖਾਨਾ_tiles, heads[h].v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase1b_ਕਾਮਾ, &bctx, n_batch);

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][1], heads[h].q_ਪੰਕਤੀ_tiles, heads[h].k_ਖਾਨਾ_tiles, heads[h].qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][3], heads[h].q_ਅਭਾਜ_ਪੰਕਤੀ_tiles, heads[h].s_ਖਾਨਾ_tiles, heads[h].o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase2_ਕਾਮਾ, &bctx, n_batch);

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][7],
                    (__fp16 *) heads[h].vtcm_m,
                    heads[h].inv_ਪੰਕਤੀ_tiles + 0 * HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    1, 1, 1, vtcm_scales_1
                );
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][8],
                    (__fp16 *) heads[h].vtcm_m + HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    heads[h].inv_ਪੰਕਤੀ_tiles + 2 * HMX_FP16_TILE_N_ELMS,
                    1, 1, 1, vtcm_scales_1
                );
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase3_ਕਾਮਾ, &bctx, n_batch);

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][4], heads[h].inv_ਪੰਕਤੀ_tiles, heads[h].v_ਅਭਾਜ_ਖਾਨਾ_tiles, heads[h].delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase4_ਕਾਮਾ, &bctx, n_batch);

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][5], heads[h].a_ਪੰਕਤੀ_tiles, heads[h].delta_ਖਾਨਾ_tiles, heads[h].o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ਧੱਕੋ_hmx_gemm_task(hmx_q, &gemm_tasks[h][6], heads[h].d_ਪੰਕਤੀ_tiles, heads[h].k_ਖਾਨਾ_tiles_64x128, heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);
            }

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase5_ਕਾਮਾ, &bctx, n_batch);

            ਅਚਲ uint32_t valid_ਟੋਕਨ = hex_smin(chunk_ਮਾਪ, n_ਟੋਕਨ - t_chunk);
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                ਅਚਲ dma_addr_t attn_chunk_dma = dst->ਅੰਕੜੇ +
                    ((uint64_t) heads[h].iv3 * n_ਟੋਕਨ * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), valid_ਟੋਕਨ);
            }

            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, gdn_hvx_phase6_ਕਾਮਾ, &bctx, n_batch);
        }

        ਜੇ (n_chunks > 0) {
            ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
                dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
            }
        }

        ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(heads[h].state_ਬਾਹਰਲਾ_dma, heads[h].s_state),
                           S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), S_v);
        }
        ਲਈ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_gated_delta_net(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * q     = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k     = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v     = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * g     = octx->src[3];
    ਅਚਲ ਬਣਤਰ htp_tensor * beta  = octx->src[4];
    ਅਚਲ ਬਣਤਰ htp_tensor * state = octx->src[5];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst   = octx->dst;

    ਜੇ (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t S_v      = v->ne[0];
    ਅਚਲ uint32_t H        = v->ne[1];
    ਅਚਲ uint32_t n_ਟੋਕਨ = v->ne[2];
    ਅਚਲ uint32_t n_seqs   = v->ne[3];
    ਅਚਲ uint32_t K        = octx->op_params[0];

    ਜੇ (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_ਟੋਕਨ == 0 || n_seqs == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    ਜੇ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    ਜੇ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_ਟੋਕਨ || k->ne[2] != n_ਟੋਕਨ || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    ਜੇ (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    ਜੇ (dst->ne[0] != S_v * H || dst->ne[1] != n_ਟੋਕਨ * n_seqs + S_v * n_seqs * K) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 5; i++) {
        ਜੇ (htp_tensor_is_extended(octx->src[i])) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }
    ਜੇ (htp_tensor_is_extended(octx->dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    ਜੇ (octx->dsts[1]) {
        ਅਚਲ ਬਣਤਰ htp_tensor * dst_cache = octx->dsts[1];
        ਜੇ (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params *) octx->kernel_params;
    ਬਣਤਰ htp_gdn_kernel_params kparams_local;
    ਜੇ (!kparams || kparams->S_v == 0) {
        ਅਚਲ uint32_t rq3 = n_seqs / q->ne[3];
        ਅਚਲ uint32_t rk3 = n_seqs / k->ne[3];
        ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = H * n_seqs;
        uint32_t n_ਧਾਗੇ = (ਕੁੱਲ_ਪੰਕਤੀਆਂ < octx->n_ਧਾਗੇ) ? ਕੁੱਲ_ਪੰਕਤੀਆਂ : octx->n_ਧਾਗੇ;
        ਜੇ (n_ਧਾਗੇ == 0) {
            n_ਧਾਗੇ = 1;
        }

        memset(&kparams_local, 0, ਆਕਾਰ(kparams_local));
        kparams_local.n_ਧਾਗੇ           = n_ਧਾਗੇ;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_ਟੋਕਨ            = n_ਟੋਕਨ;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.ਕੁੱਲ_ਪੰਕਤੀਆਂ          = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        kparams_local.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread     = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
        ਅਚਲ bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_ਟੋਕਨ >= HTP_GDN_MIN_ਟੋਕਨ) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        ਬਣਤਰ htp_gdn_hmx_vtcm_layout hmx_layout_local;
        ਬਣਤਰ htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        ਜੇ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_ਮਾਪ, ਕੁੱਲ_ਪੰਕਤੀਆਂ, octx->ctx->vtcm_ਮਾਪ, n_ਧਾਗੇ, true, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_ਮਾਪ      = HTP_GDN_CHUNK_ਮਾਪ;
            kparams_local.n_chunks        = (n_ਟੋਕਨ + HTP_GDN_CHUNK_ਮਾਪ - 1) / HTP_GDN_CHUNK_ਮਾਪ;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_ਮਾਪ       = (uint32_t) hmx_layout_local.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_ਬਾਈਟਾਂ;
            kparams_local.vtcm_ਪ੍ਰਤੀ_thread = (uint32_t) (hmx_layout_local.ਕੁੱਲ_ਬਾਈਟਾਂ / (n_ਧਾਗੇ > 0 ? n_ਧਾਗੇ : 1));
        } ਨਹੀਂਤਾਂ {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_ਧਾਗੇ);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_ਪ੍ਰਤੀ_thread = (uint32_t) hvx_layout_local.ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
            kparams_local.vtcm_ਮਾਪ       = (uint32_t) hvx_layout_local.ਕੁੱਲ_ਬਾਈਟਾਂ;
        }
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((ਦਸ਼ਮਲਵ) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        kparams_local.state_ਮਾਪ_ਪ੍ਰਤੀ_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_ਮੁੱਲਾਂ(H);
        kparams_local.div_q1        = init_fastdiv_ਮੁੱਲਾਂ(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_ਮੁੱਲਾਂ(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_ਮੁੱਲਾਂ(rq3);
        kparams_local.div_rk3       = init_fastdiv_ਮੁੱਲਾਂ(rk3);
        kparams_local.div_n_ਧਾਗੇ = init_fastdiv_ਮੁੱਲਾਂ(n_ਧਾਗੇ);

        kparams = &kparams_local;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            ਕੁੱਲ_ਪੰਕਤੀਆਂ,
            can_ਵੰਡੋ ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.ਗਿਣਤੀ_2,
            &octx->ctx->mdev.ਗਿਣਤੀ_div
        );
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    } ਨਹੀਂਤਾਂ ਜੇ (octx->op_params[1] != 0) {
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        ਵਾਪਸ gated_delta_net_f32_hmx_chunked(octx, kparams, ਪੰਕਤੀ_ਸ਼ੁਰੂ, nrows);
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = (nrows < kparams->n_ਧਾਗੇ) ? nrows : kparams->n_ਧਾਗੇ;

    ਬਣਤਰ htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.ਪੰਕਤੀ_ਸ਼ੁਰੂ = ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    gctx.nrows     = nrows;
    gctx.vtcm_ਅਧਾਰ = octx->ctx->vtcm_ਅਧਾਰ;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_ਧਾਗੇ);

    ਜੇ (gctx.layout.ਕੁੱਲ_ਬਾਈਟਾਂ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    FARF(ਉੱਚਾ_2, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.ਕੁੱਲ_ਬਾਈਟਾਂ, n_ਧਾਗੇ);

    ਜੇ (n_ਟੋਕਨ == 1) {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, gated_delta_net_f32_tg_thread, &gctx, n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, gated_delta_net_f32_pp_thread, &gctx, n_ਧਾਗੇ);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
