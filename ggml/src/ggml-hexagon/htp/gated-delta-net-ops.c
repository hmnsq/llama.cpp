#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "hvx-reduce.h"
#ಅನ್ತರ್ಭಾವ "hvx-exp.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "hmx-utils.h"
#ಅನ್ತರ್ಭಾವ "hmx-fa-kernels.h"
#ಅನ್ತರ್ಭಾವ "hmx-queue.h"
#ಅನ್ತರ್ಭಾವ "gated-delta-net-ops.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MIN
#ವ್ಯಾಖ್ಯಾನಿಸು MIN(a, b) ((a) < (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_gdn_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams;
    ರಚನೆ htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_ತಳ;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t nrows;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_mul_dot_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ mul, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vd   = hvx_vmemu(dst + i * epv);
        HVX_ಸದಿಶ vm   = mul[i];
        HVX_ಸದಿಶ vdot = dot[i];
        HVX_ಸದಿಶ ಹೊರಹೋಗು  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = ಹೊರಹೋಗು;
        acc = hvx_vec_ಕೂಡಿಸು_f32_f32(acc, hvx_vec_mul_f32_f32(ಹೊರಹೋಗು, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vm = mul[nvec];
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ ಹೊರಹೋಗು = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಹೊರಹೋಗು);
        acc = hvx_vec_ಕೂಡಿಸು_f32_f32(acc, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(ಹೊರಹೋಗು, vdot), zero));
    }

    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32(acc);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_mul_scalar_dot_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst, HVX_ಸದಿಶ vmul, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vd   = hvx_vmemu(dst + i * epv);
        HVX_ಸದಿಶ vdot = dot[i];
        HVX_ಸದಿಶ ಹೊರಹೋಗು  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = ಹೊರಹೋಗು;
        acc = hvx_vec_ಕೂಡಿಸು_f32_f32(acc, hvx_vec_mul_f32_f32(ಹೊರಹೋಗು, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ ಹೊರಹೋಗು = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಹೊರಹೋಗು);
        acc = hvx_vec_ಕೂಡಿಸು_f32_f32(acc, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(ಹೊರಹೋಗು, vdot), zero));
    }

    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32(acc);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_ಕೂಡಿಸು_scaled_dot_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ src,
        HVX_ಸದಿಶ vscale, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vd   = hvx_vmemu(dst + i * epv);
        HVX_ಸದಿಶ vs   = src[i];
        HVX_ಸದಿಶ vdot = dot[i];
        HVX_ಸದಿಶ ಹೊರಹೋಗು  = hvx_vec_ಕೂಡಿಸು_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = ಹೊರಹೋಗು;
        acc = hvx_vec_ಕೂಡಿಸು_f32_f32(acc, hvx_vec_mul_f32_f32(ಹೊರಹೋಗು, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vs   = src[nvec];
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ ಹೊರಹೋಗು = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಹೊರಹೋಗು);
        acc = hvx_vec_ಕೂಡಿಸು_f32_f32(acc, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(ಹೊರಹೋಗು, vdot), zero));
    }

    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32(acc);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_mul_dot4_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst0, ದಶಮಾಂಶ * ಸೀಮಿತ dst1,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst2, ದಶಮಾಂಶ * ಸೀಮಿತ dst3,
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ mul, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc0 = Q6_V_vzero();
    HVX_ಸದಿಶ acc1 = Q6_V_vzero();
    HVX_ಸದಿಶ acc2 = Q6_V_vzero();
    HVX_ಸದಿಶ acc3 = Q6_V_vzero();

    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vm   = mul[i];
        HVX_ಸದಿಶ vdot = dot[i];

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vm = mul[nvec];
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out3);

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ಸದಿಶ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32x4(acc);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_mul_scalar_dot4_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst0, ದಶಮಾಂಶ * ಸೀಮಿತ dst1,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst2, ದಶಮಾಂಶ * ಸೀಮಿತ dst3,
        HVX_ಸದಿಶ vmul, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc0 = Q6_V_vzero();
    HVX_ಸದಿಶ acc1 = Q6_V_vzero();
    HVX_ಸದಿಶ acc2 = Q6_V_vzero();
    HVX_ಸದಿಶ acc3 = Q6_V_vzero();

    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vdot = dot[i];

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out3);

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ಸದಿಶ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32x4(acc);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_ಕೂಡಿಸು_scaled_dot4_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst0, ದಶಮಾಂಶ * ಸೀಮಿತ dst1,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst2, ದಶಮಾಂಶ * ಸೀಮಿತ dst3,
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ scale,
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc0 = Q6_V_vzero();
    HVX_ಸದಿಶ acc1 = Q6_V_vzero();
    HVX_ಸದಿಶ acc2 = Q6_V_vzero();
    HVX_ಸದಿಶ acc3 = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale0 = hvx_vec_splat_f32(scale[0]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale1 = hvx_vec_splat_f32(scale[1]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale2 = hvx_vec_splat_f32(scale[2]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale3 = hvx_vec_splat_f32(scale[3]);

    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vs   = src[i];
        HVX_ಸದಿಶ vdot = dot[i];

        HVX_ಸದಿಶ out0 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ಸದಿಶ out1 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ಸದಿಶ out2 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ಸದಿಶ out3 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vs   = src[nvec];
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ out0 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ಸದಿಶ out1 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ಸದಿಶ out2 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ಸದಿಶ out3 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out3);

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_ಸದಿಶ_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32x4(acc);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_mul_dot8_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst0, ದಶಮಾಂಶ * ಸೀಮಿತ dst1,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst2, ದಶಮಾಂಶ * ಸೀಮಿತ dst3, ದಶಮಾಂಶ * ಸೀಮಿತ dst4,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst5, ದಶಮಾಂಶ * ಸೀಮಿತ dst6, ದಶಮಾಂಶ * ಸೀಮಿತ dst7,
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ mul, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc0 = Q6_V_vzero();
    HVX_ಸದಿಶ acc1 = Q6_V_vzero();
    HVX_ಸದಿಶ acc2 = Q6_V_vzero();
    HVX_ಸದಿಶ acc3 = Q6_V_vzero();
    HVX_ಸದಿಶ acc4 = Q6_V_vzero();
    HVX_ಸದಿಶ acc5 = Q6_V_vzero();
    HVX_ಸದಿಶ acc6 = Q6_V_vzero();
    HVX_ಸದಿಶ acc7 = Q6_V_vzero();

    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vm   = mul[i];
        HVX_ಸದಿಶ vdot = dot[i];

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_ಸದಿಶ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_ಸದಿಶ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_ಸದಿಶ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_ಸದಿಶ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vm = mul[nvec];
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_ಸದಿಶ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_ಸದಿಶ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_ಸದಿಶ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_ಸದಿಶ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out3);
        hvx_vec_store_u(dst4 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out4);
        hvx_vec_store_u(dst5 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out5);
        hvx_vec_store_u(dst6 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out6);
        hvx_vec_store_u(dst7 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out7);

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc4, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc5, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc6, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc7, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ಸದಿಶ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ಸದಿಶ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ಸದಿಶ rA = hvx_vec_reduce_ಮೊತ್ತ_f32x4(accA);
    HVX_ಸದಿಶ rB = hvx_vec_reduce_ಮೊತ್ತ_f32x4(accB);
    HVX_ಸದಿಶ_Pred q16 = Q6_Q_vsetq2_R(16);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_mul_scalar_dot8_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst0, ದಶಮಾಂಶ * ಸೀಮಿತ dst1,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst2, ದಶಮಾಂಶ * ಸೀಮಿತ dst3, ದಶಮಾಂಶ * ಸೀಮಿತ dst4,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst5, ದಶಮಾಂಶ * ಸೀಮಿತ dst6, ದಶಮಾಂಶ * ಸೀಮಿತ dst7,
        HVX_ಸದಿಶ vmul, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc0 = Q6_V_vzero();
    HVX_ಸದಿಶ acc1 = Q6_V_vzero();
    HVX_ಸದಿಶ acc2 = Q6_V_vzero();
    HVX_ಸದಿಶ acc3 = Q6_V_vzero();
    HVX_ಸದಿಶ acc4 = Q6_V_vzero();
    HVX_ಸದಿಶ acc5 = Q6_V_vzero();
    HVX_ಸದಿಶ acc6 = Q6_V_vzero();
    HVX_ಸದಿಶ acc7 = Q6_V_vzero();

    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vdot = dot[i];

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_ಸದಿಶ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_ಸದಿಶ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_ಸದಿಶ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_ಸದಿಶ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_ಸದಿಶ out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_ಸದಿಶ out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_ಸದಿಶ out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_ಸದಿಶ out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_ಸದಿಶ out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_ಸದಿಶ out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_ಸದಿಶ out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out3);
        hvx_vec_store_u(dst4 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out4);
        hvx_vec_store_u(dst5 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out5);
        hvx_vec_store_u(dst6 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out6);
        hvx_vec_store_u(dst7 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out7);

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc4, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc5, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc6, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc7, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ಸದಿಶ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ಸದಿಶ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ಸದಿಶ rA = hvx_vec_reduce_ಮೊತ್ತ_f32x4(accA);
    HVX_ಸದಿಶ rB = hvx_vec_reduce_ಮೊತ್ತ_f32x4(accB);
    HVX_ಸದಿಶ_Pred q16 = Q6_Q_vsetq2_R(16);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ gdn_ಕೂಡಿಸು_scaled_dot8_f32(ದಶಮಾಂಶ * ಸೀಮಿತ dst0, ದಶಮಾಂಶ * ಸೀಮಿತ dst1,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst2, ದಶಮಾಂಶ * ಸೀಮಿತ dst3, ದಶಮಾಂಶ * ಸೀಮಿತ dst4,
        ದಶಮಾಂಶ * ಸೀಮಿತ dst5, ದಶಮಾಂಶ * ಸೀಮಿತ dst6, ದಶಮಾಂಶ * ಸೀಮಿತ dst7,
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ scale,
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ dot, uint32_t n) {
    HVX_ಸದಿಶ acc0 = Q6_V_vzero();
    HVX_ಸದಿಶ acc1 = Q6_V_vzero();
    HVX_ಸದಿಶ acc2 = Q6_V_vzero();
    HVX_ಸದಿಶ acc3 = Q6_V_vzero();
    HVX_ಸದಿಶ acc4 = Q6_V_vzero();
    HVX_ಸದಿಶ acc5 = Q6_V_vzero();
    HVX_ಸದಿಶ acc6 = Q6_V_vzero();
    HVX_ಸದಿಶ acc7 = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale0 = hvx_vec_splat_f32(scale[0]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale1 = hvx_vec_splat_f32(scale[1]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale2 = hvx_vec_splat_f32(scale[2]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale3 = hvx_vec_splat_f32(scale[3]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale4 = hvx_vec_splat_f32(scale[4]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale5 = hvx_vec_splat_f32(scale[5]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale6 = hvx_vec_splat_f32(scale[6]);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale7 = hvx_vec_splat_f32(scale[7]);

    ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;
    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        HVX_ಸದಿಶ vs   = src[i];
        HVX_ಸದಿಶ vdot = dot[i];

        HVX_ಸದಿಶ out0 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ಸದಿಶ out1 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ಸದಿಶ out2 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ಸದಿಶ out3 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_ಸದಿಶ out4 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_ಸದಿಶ out5 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_ಸದಿಶ out6 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_ಸದಿಶ out7 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    ಆದರೆ (nloe) {
        ಸ್ಥಿರಾಂಕ uint32_t off = nvec * epv;
        HVX_ಸದಿಶ vs   = src[nvec];
        HVX_ಸದಿಶ vdot = dot[nvec];
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ zero = Q6_V_vzero();

        HVX_ಸದಿಶ out0 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_ಸದಿಶ out1 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_ಸದಿಶ out2 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_ಸದಿಶ out3 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_ಸದಿಶ out4 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_ಸದಿಶ out5 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_ಸದಿಶ out6 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_ಸದಿಶ out7 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out0);
        hvx_vec_store_u(dst1 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out1);
        hvx_vec_store_u(dst2 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out2);
        hvx_vec_store_u(dst3 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out3);
        hvx_vec_store_u(dst4 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out4);
        hvx_vec_store_u(dst5 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out5);
        hvx_vec_store_u(dst6 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out6);
        hvx_vec_store_u(dst7 + off, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), out7);

        acc0 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc0, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc1, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc2, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc3, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc4, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc5, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc6, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_ಕೂಡಿಸು_f32_f32(acc7, Q6_V_vmux_QVV(ಮುಖವಾಡ, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_ಸದಿಶ_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_ಸದಿಶ_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_ಸದಿಶ rA = hvx_vec_reduce_ಮೊತ್ತ_f32x4(accA);
    HVX_ಸದಿಶ rB = hvx_vec_reduce_ಮೊತ್ತ_f32x4(accB);
    HVX_ಸದಿಶ_Pred q16 = Q6_Q_vsetq2_R(16);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_ಹೆಜ್ಜೆ_kda_f32(
    ದಶಮಾಂಶ * ಸೀಮಿತ s_work,
    ದಶಮಾಂಶ * ಸೀಮಿತ attn_ಹೊರಹೋಗು,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ q_t,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ k_t,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ v_t,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ g_t,
    ದಶಮಾಂಶ beta_val,
    ದಶಮಾಂಶ scale,
    uint32_t S_v
) {
    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = S_v / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = S_v % epv;

    HVX_ಸದಿಶ vq[4];
    HVX_ಸದಿಶ vk[4];
    HVX_ಸದಿಶ vg[4];

    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    ಆದರೆ (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
    }

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vbeta  = hvx_vec_splat_f32(beta_val);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vscale = hvx_vec_splat_f32(scale);

    ದಶಮಾಂಶ delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    ಗಾಗಿ (; j + 8 <= S_v; j += 8) {
        ದಶಮಾಂಶ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ದಶಮಾಂಶ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ದಶಮಾಂಶ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ದಶಮಾಂಶ * row3 = s_work + (uint64_t) (j + 3) * S_v;
        ದಶಮಾಂಶ * row4 = s_work + (uint64_t) (j + 4) * S_v;
        ದಶಮಾಂಶ * row5 = s_work + (uint64_t) (j + 5) * S_v;
        ದಶಮಾಂಶ * row6 = s_work + (uint64_t) (j + 6) * S_v;
        ದಶಮಾಂಶ * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_ಸದಿಶ vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_ಸದಿಶ vv_t   = hvx_vmemu(v_t + j);
        HVX_ಸದಿಶ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ಸದಿಶ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * ಗಾತ್ರ(ದಶಮಾಂಶ), vdelta);

        HVX_ಸದಿಶ vattn = gdn_ಕೂಡಿಸು_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_ಸದಿಶ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ಹೊರಹೋಗು + j, 8 * ಗಾತ್ರ(ದಶಮಾಂಶ), res_attn);
    }
    ಗಾಗಿ (; j + 4 <= S_v; j += 4) {
        ದಶಮಾಂಶ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ದಶಮಾಂಶ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ದಶಮಾಂಶ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ದಶಮಾಂಶ * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_ಸದಿಶ vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_ಸದಿಶ vv_t   = hvx_vmemu(v_t + j);
        HVX_ಸದಿಶ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ಸದಿಶ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * ಗಾತ್ರ(ದಶಮಾಂಶ), vdelta);

        HVX_ಸದಿಶ vattn = gdn_ಕೂಡಿಸು_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_ಸದಿಶ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ಹೊರಹೋಗು + j, 4 * ಗಾತ್ರ(ದಶಮಾಂಶ), res_attn);
    }
    ಗಾಗಿ (; j < S_v; ++j) {
        ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = s_work + (uint64_t) j * S_v;
        HVX_ಸದಿಶ vsum = gdn_mul_dot_f32(ಅಡ್ಡಸಾಲು, vg, vk, S_v);
        HVX_ಸದಿಶ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_ಸದಿಶ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_ಸದಿಶ vres = gdn_ಕೂಡಿಸು_scaled_dot_f32(ಅಡ್ಡಸಾಲು, vk, vdj, vq, S_v);
        attn_ಹೊರಹೋಗು[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_ಹೆಜ್ಜೆ_scalar_f32(
    ದಶಮಾಂಶ * ಸೀಮಿತ s_work,
    ದಶಮಾಂಶ * ಸೀಮಿತ attn_ಹೊರಹೋಗು,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ q_t,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ k_t,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ v_t,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ g_t,
    ದಶಮಾಂಶ beta_val,
    ದಶಮಾಂಶ scale,
    uint32_t S_v
) {
    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = S_v / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = S_v % epv;

    HVX_ಸದಿಶ vq[4];
    HVX_ಸದಿಶ vk[4];

    ಗಾಗಿ (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    ಆದರೆ (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vbeta  = hvx_vec_splat_f32(beta_val);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vscale = hvx_vec_splat_f32(scale);

    ದಶಮಾಂಶ delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    ಗಾಗಿ (; j + 8 <= S_v; j += 8) {
        ದಶಮಾಂಶ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ದಶಮಾಂಶ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ದಶಮಾಂಶ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ದಶಮಾಂಶ * row3 = s_work + (uint64_t) (j + 3) * S_v;
        ದಶಮಾಂಶ * row4 = s_work + (uint64_t) (j + 4) * S_v;
        ದಶಮಾಂಶ * row5 = s_work + (uint64_t) (j + 5) * S_v;
        ದಶಮಾಂಶ * row6 = s_work + (uint64_t) (j + 6) * S_v;
        ದಶಮಾಂಶ * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_ಸದಿಶ vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_ಸದಿಶ vv_t   = hvx_vmemu(v_t + j);
        HVX_ಸದಿಶ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ಸದಿಶ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * ಗಾತ್ರ(ದಶಮಾಂಶ), vdelta);

        HVX_ಸದಿಶ vattn = gdn_ಕೂಡಿಸು_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_ಸದಿಶ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ಹೊರಹೋಗು + j, 8 * ಗಾತ್ರ(ದಶಮಾಂಶ), res_attn);
    }
    ಗಾಗಿ (; j + 4 <= S_v; j += 4) {
        ದಶಮಾಂಶ * row0 = s_work + (uint64_t) (j + 0) * S_v;
        ದಶಮಾಂಶ * row1 = s_work + (uint64_t) (j + 1) * S_v;
        ದಶಮಾಂಶ * row2 = s_work + (uint64_t) (j + 2) * S_v;
        ದಶಮಾಂಶ * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_ಸದಿಶ vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_ಸದಿಶ vv_t   = hvx_vmemu(v_t + j);
        HVX_ಸದಿಶ diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_ಸದಿಶ vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * ಗಾತ್ರ(ದಶಮಾಂಶ), vdelta);

        HVX_ಸದಿಶ vattn = gdn_ಕೂಡಿಸು_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_ಸದಿಶ res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_ಹೊರಹೋಗು + j, 4 * ಗಾತ್ರ(ದಶಮಾಂಶ), res_attn);
    }
    ಗಾಗಿ (; j < S_v; ++j) {
        ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = s_work + (uint64_t) j * S_v;
        HVX_ಸದಿಶ vsum = gdn_mul_scalar_dot_f32(ಅಡ್ಡಸಾಲು, vgate, vk, S_v);
        HVX_ಸದಿಶ vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_ಸದಿಶ vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_ಸದಿಶ vres = gdn_ಕೂಡಿಸು_scaled_dot_f32(ಅಡ್ಡಸಾಲು, vk, vdj, vq, S_v);
        attn_ಹೊರಹೋಗು[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gated_delta_net_f32_pp_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_gdn_context * gctx = (ರಚನೆ htp_gdn_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = gctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams = gctx->kparams;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q     = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k     = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v     = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * g     = octx->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * beta  = octx->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * state = octx->src[5];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst   = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t S_v      = kparams->S_v;
    ಸ್ಥಿರಾಂಕ uint32_t H        = kparams->H;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = kparams->n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t n_seqs   = kparams->n_seqs;
    ಸ್ಥಿರಾಂಕ uint32_t K        = kparams->K;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ  = gctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + gctx->nrows;

    ಆದರೆ (ith >= gctx->nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst_cache = octx->dsts[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = kparams->scale;
    ದಶಮಾಂಶ * dst_ತಳ       = (ದಶಮಾಂಶ *) (uintptr_t) dst->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * state_ಹೊರಹೋಗು_ತಳ = dst_cache ? (ದಶಮಾಂಶ *) (uintptr_t) dst_cache->ದತ್ತಾಂಶ : (dst_ತಳ + S_v * H * n_ಟೋಕನುಗಳು * n_seqs);

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_vtcm_layout * layout = &gctx->layout;
    ದಶಮಾಂಶ * s_work[2];
    s_work[0] = (ದಶಮಾಂಶ *) (gctx->vtcm_ತಳ + layout->ಬೈಟುಗಳು_ಪ್ರತಿ_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_H   = &kparams->div_H;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_q1  = &kparams->div_q1;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_k1  = &kparams->div_k1;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_rq3 = &kparams->div_rq3;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_rk3 = &kparams->div_rk3;

    ಸ್ಥಿರಾಂಕ uint32_t state_seq_stride = kparams->state_seq_stride;
    ಸ್ಥಿರಾಂಕ uint64_t state_ಗಾತ್ರ_ಪ್ರತಿ_snap = (uint64_t) kparams->state_ಗಾತ್ರ_ಪ್ರತಿ_snap;
    ಸ್ಥಿರಾಂಕ dma_addr_t state_ಹೊರಹೋಗು_dma_ತಳ = dst_cache ? dst_cache->ದತ್ತಾಂಶ : (dst->ದತ್ತಾಂಶ + S_v * H * n_ಟೋಕನುಗಳು * n_seqs * ಗಾತ್ರ(ದಶಮಾಂಶ));

    uint32_t ir_prefetch = gctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < 2 && ir_prefetch < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ಹೆಜ್ಜೆ++) {
        ಸ್ಥಿರಾಂಕ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ಸ್ಥಿರಾಂಕ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_ಒಳ  = state->ದತ್ತಾಂಶ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ಗಾತ್ರ(ದಶಮಾಂಶ);
        dma_addr_t ps_ಹೊರಹೋಗು = state_ಹೊರಹೋಗು_dma_ತಳ + ((uint64_t) piv3 * H + piv1) * S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ);

        // Push dummy write-back
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ps_ಹೊರಹೋಗು, s_work[spad_idx]),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), 0);

        // Push fetch
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_work[spad_idx], ps_ಒಳ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಪೂರ್ಣಾಂಕ curr_spad_idx = 0;
    ಗಾಗಿ (uint32_t ir = gctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith; ir < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ir += nth) {
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);

        ದಶಮಾಂಶ * s_work_curr = s_work[curr_spad_idx];

        ಸ್ಥಿರಾಂಕ uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ಸ್ಥಿರಾಂಕ uint32_t iv3 = fastdiv(ir, fd_H);

        ಸ್ಥಿರಾಂಕ uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ಸ್ಥಿರಾಂಕ uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ಸ್ಥಿರಾಂಕ uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ಸ್ಥಿರಾಂಕ uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_ಹೊರಹೋಗು  = state_ಹೊರಹೋಗು_dma_ತಳ + ((uint64_t) iv3 * H + iv1) * S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ದಶಮಾಂಶ * attn_ದತ್ತಾಂಶ = dst_ತಳ + ((uint64_t) iv3 * n_ಟೋಕನುಗಳು * H + iv1) * S_v;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ಗಾಗಿ (uint32_t t = 0; t < n_ಟೋಕನುಗಳು; ++t) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) q->ದತ್ತಾಂಶ +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) k->ದತ್ತಾಂಶ +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) v->ದತ್ತಾಂಶ +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) g->ದತ್ತಾಂಶ +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) beta->ದತ್ತಾಂಶ +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            ಆದರೆ (kparams->kda) {
                gdn_ಹೆಜ್ಜೆ_kda_f32(s_work_curr, attn_ದತ್ತಾಂಶ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } ಇಲ್ಲದಿದ್ದರೆ {
                gdn_ಹೆಜ್ಜೆ_scalar_f32(s_work_curr, attn_ದತ್ತಾಂಶ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            ಆದರೆ (K > 1) {
                ಸ್ಥಿರಾಂಕ int64_t ಗುರಿ_ಸ್ಥಾನ = (int64_t) n_ಟೋಕನುಗಳು - 1 - (int64_t) t;
                ಆದರೆ (ಗುರಿ_ಸ್ಥಾನ > 0 && ಗುರಿ_ಸ್ಥಾನ < (int64_t) K) {
                    ದಶಮಾಂಶ * curr_state_o = state_ಹೊರಹೋಗು_ತಳ + (uint64_t) ಗುರಿ_ಸ್ಥಾನ * state_ಗಾತ್ರ_ಪ್ರತಿ_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (ಸ್ಥಿರಾಂಕ uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_ದತ್ತಾಂಶ += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_ಹೊರಹೋಗು, s_work_curr),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);

        // Prefetch next block (if any)
        ಆದರೆ (ir_prefetch < ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ಸ್ಥಿರಾಂಕ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_ಒಳ = state->ದತ್ತಾಂಶ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ಗಾತ್ರ(ದಶಮಾಂಶ);

            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_work[spad_idx], ps_ಒಳ),
                           S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                           S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಶೂನ್ಯ gated_delta_net_f32_tg_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_gdn_context * gctx = (ರಚನೆ htp_gdn_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = gctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams = gctx->kparams;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q     = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k     = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v     = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * g     = octx->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * beta  = octx->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * state = octx->src[5];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst   = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t S_v      = kparams->S_v;
    ಸ್ಥಿರಾಂಕ uint32_t H        = kparams->H;
    ಸ್ಥಿರಾಂಕ uint32_t n_seqs   = kparams->n_seqs;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ  = gctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + gctx->nrows;

    ಆದರೆ (ith >= gctx->nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst_cache = octx->dsts[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = kparams->scale;
    ದಶಮಾಂಶ * dst_ತಳ  = (ದಶಮಾಂಶ *) (uintptr_t) dst->ದತ್ತಾಂಶ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_vtcm_layout * layout = &gctx->layout;
    ದಶಮಾಂಶ * s_work[2];
    s_work[0] = (ದಶಮಾಂಶ *) (gctx->vtcm_ತಳ + layout->ಬೈಟುಗಳು_ಪ್ರತಿ_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_H   = &kparams->div_H;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_q1  = &kparams->div_q1;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_k1  = &kparams->div_k1;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_rq3 = &kparams->div_rq3;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * fd_rk3 = &kparams->div_rk3;

    ಸ್ಥಿರಾಂಕ uint32_t state_seq_stride = kparams->state_seq_stride;
    ಸ್ಥಿರಾಂಕ dma_addr_t state_ಹೊರಹೋಗು_dma_ತಳ = dst_cache ? dst_cache->ದತ್ತಾಂಶ : (dst->ದತ್ತಾಂಶ + S_v * H * n_seqs * ಗಾತ್ರ(ದಶಮಾಂಶ));

    uint32_t ir_prefetch = gctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < 2 && ir_prefetch < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ಹೆಜ್ಜೆ++) {
        ಸ್ಥಿರಾಂಕ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        ಸ್ಥಿರಾಂಕ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_ಒಳ  = state->ದತ್ತಾಂಶ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ಗಾತ್ರ(ದಶಮಾಂಶ);
        dma_addr_t ps_ಹೊರಹೋಗು = state_ಹೊರಹೋಗು_dma_ತಳ + ((uint64_t) piv3 * H + piv1) * S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ);

        // Push dummy write-back
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ps_ಹೊರಹೋಗು, s_work[spad_idx]),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), 0);

        // Push fetch
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_work[spad_idx], ps_ಒಳ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಪೂರ್ಣಾಂಕ curr_spad_idx = 0;
    ಗಾಗಿ (uint32_t ir = gctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith; ir < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ir += nth) {
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);

        ದಶಮಾಂಶ * s_work_curr = s_work[curr_spad_idx];

        ಸ್ಥಿರಾಂಕ uint32_t iv1 = fastmodulo(ir, H, fd_H);
        ಸ್ಥಿರಾಂಕ uint32_t iv3 = fastdiv(ir, fd_H);

        ಸ್ಥಿರಾಂಕ uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        ಸ್ಥಿರಾಂಕ uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        ಸ್ಥಿರಾಂಕ uint32_t iq3 = fastdiv(iv3, fd_rq3);
        ಸ್ಥಿರಾಂಕ uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_ಹೊರಹೋಗು  = state_ಹೊರಹೋಗು_dma_ತಳ + ((uint64_t) iv3 * H + iv1) * S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ದಶಮಾಂಶ * attn_ದತ್ತಾಂಶ = dst_ತಳ + ((uint64_t) iv3 * H + iv1) * S_v;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) q->ದತ್ತಾಂಶ +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) k->ದತ್ತಾಂಶ +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) v->ದತ್ತಾಂಶ +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) g->ದತ್ತಾಂಶ +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) beta->ದತ್ತಾಂಶ +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ಆದರೆ (kparams->kda) {
            gdn_ಹೆಜ್ಜೆ_kda_f32(s_work_curr, attn_ದತ್ತಾಂಶ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } ಇಲ್ಲದಿದ್ದರೆ {
            gdn_ಹೆಜ್ಜೆ_scalar_f32(s_work_curr, attn_ದತ್ತಾಂಶ, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_ಹೊರಹೋಗು, s_work_curr),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                       S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);

        // Prefetch next block (if any)
        ಆದರೆ (ir_prefetch < ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            ಸ್ಥಿರಾಂಕ uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_ಒಳ = state->ದತ್ತಾಂಶ + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * ಗಾತ್ರ(ದಶಮಾಂಶ);

            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_work[spad_idx], ps_ಒಳ),
                           S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ),
                           S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_ಸರತಿ_flush(dma_q);
}

ರಚನೆ htp_gdn_hmx_gemm_task {
    ಸ್ಥಿರಾಂಕ __fp16 * ಅಡ್ಡಸಾಲು_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * ಲಂಬಸಾಲು_tiles;
    __fp16 *       ಹೊರಹೋಗು_tiles;
    uint32_t       n_ಅಡ್ಡಸಾಲು_tiles;
    uint32_t       n_ಲಂಬಸಾಲು_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_gdn_hmx_gemm_ಕೆಲಸಗಾರ(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_gdn_hmx_gemm_task * task = (ರಚನೆ htp_gdn_hmx_gemm_task *) ದತ್ತಾಂಶ;
    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)task->hmx_scales));

    ಸ್ಥಿರಾಂಕ size_t dot_stride = task->dot_stride;
    ಗಾಗಿ (uint32_t r = 0; r < task->n_ಅಡ್ಡಸಾಲು_tiles; ++r) {
        ಸ್ಥಿರಾಂಕ __fp16 * r_tiles = task->ಅಡ್ಡಸಾಲು_tiles + r * dot_stride;
        ಸ್ಥಿರಾಂಕ __fp16 * c_tiles = task->ಲಂಬಸಾಲು_tiles;
        __fp16 *       o_tile  = task->ಹೊರಹೋಗು_tiles + r * task->n_ಲಂಬಸಾಲು_tiles * HMX_FP16_TILE_N_ELMS;

        ಗಾಗಿ (uint32_t c = 0; c < task->n_ಲಂಬಸಾಲು_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_gdn_ತಳ್ಳು_hmx_gemm_task(
    hmx_ಸರತಿ_t q,
    ರಚನೆ htp_gdn_hmx_gemm_task * task,
    ಸ್ಥಿರಾಂಕ __fp16 * ಅಡ್ಡಸಾಲು_tiles,
    ಸ್ಥಿರಾಂಕ __fp16 * ಲಂಬಸಾಲು_tiles,
    __fp16 * ಹೊರಹೋಗು_tiles,
    uint32_t n_ಅಡ್ಡಸಾಲು_tiles,
    uint32_t n_ಲಂಬಸಾಲು_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    task->ಅಡ್ಡಸಾಲು_tiles   = ಅಡ್ಡಸಾಲು_tiles;
    task->ಲಂಬಸಾಲು_tiles   = ಲಂಬಸಾಲು_tiles;
    task->ಹೊರಹೋಗು_tiles   = ಹೊರಹೋಗು_tiles;
    task->n_ಅಡ್ಡಸಾಲು_tiles = n_ಅಡ್ಡಸಾಲು_tiles;
    task->n_ಲಂಬಸಾಲು_tiles = n_ಲಂಬಸಾಲು_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    task->hmx_scales  = scales;

    hmx_ಸರತಿ_ತಳ್ಳು(q, hmx_ಸರತಿ_make_desc(htp_gdn_hmx_gemm_ಕೆಲಸಗಾರ, task));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_unpack_64x64_tiles_ಗಮ್ಯ_vectors(
    HVX_ಸದಿಶ * ಸೀಮಿತ ಅಡ್ಡಸಾಲುಗಳು,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ tiles
) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t00 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t01 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t10 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t11 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        HVX_ಸದಿಶ_Pair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        ಅಡ್ಡಸಾಲುಗಳು[2 * r + 0] = Q6_V_lo_W(vp0);
        ಅಡ್ಡಸಾಲುಗಳು[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_ಸದಿಶ_Pair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        ಅಡ್ಡಸಾಲುಗಳು[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        ಅಡ್ಡಸಾಲುಗಳು[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_pack_64x64_vectors_ಗಮ್ಯ_tiles(
    __fp16 * ಸೀಮಿತ tiles,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ ಅಡ್ಡಸಾಲುಗಳು
) {
    HVX_ಸದಿಶ * t00 = (HVX_ಸದಿಶ *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_ಸದಿಶ * t01 = (HVX_ಸದಿಶ *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_ಸದಿಶ * t10 = (HVX_ಸದಿಶ *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_ಸದಿಶ * t11 = (HVX_ಸದಿಶ *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        HVX_ಸದಿಶ_Pair vp0 = Q6_W_vshuff_VVR(ಅಡ್ಡಸಾಲುಗಳು[2 * r + 1], ಅಡ್ಡಸಾಲುಗಳು[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_ಸದಿಶ_Pair vp1 = Q6_W_vshuff_VVR(ಅಡ್ಡಸಾಲುಗಳು[32 + 2 * r + 1], ಅಡ್ಡಸಾಲುಗಳು[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_unpack_64x_S_tiles_ಗಮ್ಯ_f32(
    ದಶಮಾಂಶ * ಸೀಮಿತ dst_f32,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ tiles,
    uint32_t S_v
) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಲಂಬಸಾಲು_tiles = S_v / 32;
    ಗಾಗಿ (uint32_t r0 = 0; r0 < 2; ++r0) {
        ಗಾಗಿ (uint32_t d = 0; d < S_v / 64; ++d) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
                HVX_ಸದಿಶ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_ಸದಿಶ_Pair p0 = hvx_vec_f16_ಗಮ್ಯ_f32(Q6_V_lo_W(vp01));
                HVX_ಸದಿಶ_Pair p1 = hvx_vec_f16_ಗಮ್ಯ_f32(Q6_V_hi_W(vp01));

                ದಶಮಾಂಶ * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                ದಶಮಾಂಶ * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_unpack_64x_S_tiles_ಗಮ್ಯ_f16(
    __fp16 * ಸೀಮಿತ dst_f16,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ tiles,
    uint32_t S_v
) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಲಂಬಸಾಲು_tiles = S_v / 32;
    ಗಾಗಿ (uint32_t r0 = 0; r0 < 2; ++r0) {
        ಗಾಗಿ (uint32_t d = 0; d < S_v / 64; ++d) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
                HVX_ಸದಿಶ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_unpack_Sx_S_tiles_ಗಮ್ಯ_f32(
    ದಶಮಾಂಶ * ಸೀಮಿತ dst_f32,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ tiles,
    uint32_t S_v
) {
    ಸ್ಥಿರಾಂಕ uint32_t n_tiles = S_v / 32;
    ಗಾಗಿ (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        ಗಾಗಿ (uint32_t d = 0; d < S_v / 64; ++d) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * t1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
                HVX_ಸದಿಶ_Pair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_ಸದಿಶ_Pair p0 = hvx_vec_f16_ಗಮ್ಯ_f32(Q6_V_lo_W(vp01));
                HVX_ಸದಿಶ_Pair p1 = hvx_vec_f16_ಗಮ್ಯ_f32(Q6_V_hi_W(vp01));

                ದಶಮಾಂಶ * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                ದಶಮಾಂಶ * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_f32_ಗಮ್ಯ_hmx_ಅಡ್ಡಸಾಲು_tiles_and_f16(
    __fp16 * ಸೀಮಿತ dst_tiles,
    __fp16 * ಸೀಮಿತ dst_ಅವಿಭಾಜ್ಯ_tiles,
    __fp16 * ಸೀಮಿತ dst_f16,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ scale_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ
) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಲಂಬಸಾಲು_tiles = n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ / 32;
    ಸ್ಥಿರಾಂಕ uint32_t * scale_ಜೋಡಿಗಳು = (ಸ್ಥಿರಾಂಕ uint32_t *) scale_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * p0 = src + (r + 0) * n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * p1 = src + (r + 1) * n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

        HVX_ಸದಿಶ v_scale;
        ಆದರೆ (dst_ಅವಿಭಾಜ್ಯ_tiles) {
            uint32_t scale_pair = scale_ಜೋಡಿಗಳು ? scale_ಜೋಡಿಗಳು[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        ಗಾಗಿ (uint32_t c = 0; c < n_ಲಂಬಸಾಲು_tiles; c += 2) {
            HVX_ಸದಿಶ v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_ಸದಿಶ v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_ಸದಿಶ v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_ಸದಿಶ v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_ಸದಿಶ vh0 = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0_0, v1_0);
            HVX_ಸದಿಶ vh1 = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_ಸದಿಶ *) tile0)[r1] = vh0;
            ((HVX_ಸದಿಶ *) tile1)[r1] = vh1;

            ಆದರೆ (dst_ಅವಿಭಾಜ್ಯ_tiles) {
                HVX_ಸದಿಶ vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_ಸದಿಶ vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_ಅವಿಭಾಜ್ಯ_tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_ಅವಿಭಾಜ್ಯ_tiles + (r0 * n_ಲಂಬಸಾಲು_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_ಸದಿಶ *) tile0_s)[r1] = vh0_s;
                ((HVX_ಸದಿಶ *) tile1_s)[r1] = vh1_s;
            }

            ಆದರೆ (dst_f16) {
                HVX_ಸದಿಶ_Pair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_transpose_32x32_ಪದಗಳು(HVX_ಸದಿಶ * ಸೀಮಿತ m, HVX_ಸದಿಶ * ಸೀಮಿತ tmp) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 4) {
        HVX_ಸದಿಶ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ಸದಿಶ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 8) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i) {
            HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 16) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
            HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_pack_d_t_ಅಡ್ಡಸಾಲು_tiles(
    __fp16 * ಸೀಮಿತ dst_tiles,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ src_d,
    uint32_t S_v,
    HVX_ಸದಿಶ * ಸೀಮಿತ m,
    HVX_ಸದಿಶ * ಸೀಮಿತ tmp
) {
    ಗಾಗಿ (uint32_t ಲಂಬಸಾಲು_half = 0; ಲಂಬಸಾಲು_half < S_v / 64; ++ಲಂಬಸಾಲು_half) {
        uint32_t r0_ತಳ = ಲಂಬಸಾಲು_half * 2;
        ಗಾಗಿ (uint32_t c0 = 0; c0 < 2; ++c0) {
            ಗಾಗಿ (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + ಲಂಬಸಾಲು_half * 64);
            }

            hvx_transpose_32x32_ಪದಗಳು(m, tmp);

            uint32_t tile0_idx = (r0_ತಳ + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_ತಳ + 1) * 2 + c0;
            HVX_ಸದಿಶ * t0 = (HVX_ಸದಿಶ *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_ಸದಿಶ * t1 = (HVX_ಸದಿಶ *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ gdn_build_inv_l_ಖಂಡಗಳು(
    __fp16 * ಸೀಮಿತ inv_ಅಡ್ಡಸಾಲು_tiles,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ ಅಡ್ಡಸಾಲುಗಳು_kk,
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ decay_m,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ beta,
    __fp16 * ಸೀಮಿತ l10_tile,
    __fp16 * ಸೀಮಿತ neg_a11_tile
) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_one_f16 = hvx_vec_splat_f16(1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_ಸದಿಶ * ಸೀಮಿತ p_l00 = (HVX_ಸದಿಶ *) l00;
    HVX_ಸದಿಶ * ಸೀಮಿತ p_l11 = (HVX_ಸದಿಶ *) l11;
    HVX_ಸದಿಶ * ಸೀಮಿತ p_l10_tile = (HVX_ಸದಿಶ *) l10_tile;

    HVX_ಸದಿಶ * ಸೀಮಿತ tile00 = (HVX_ಸದಿಶ *) (inv_ಅಡ್ಡಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_ಸದಿಶ * ಸೀಮಿತ tile01 = (HVX_ಸದಿಶ *) (inv_ಅಡ್ಡಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_ಸದಿಶ * ಸೀಮಿತ tile11 = (HVX_ಸದಿಶ *) (inv_ಅಡ್ಡಸಾಲು_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_ಸದಿಶ * ಸೀಮಿತ p_neg_a11 = (HVX_ಸದಿಶ *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_ಗಮ್ಯ_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_ಸದಿಶ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_ಸದಿಶ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_ಸದಿಶ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_ಸದಿಶ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_ಸದಿಶ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ಅಡ್ಡಸಾಲುಗಳು_kk[t0], v_d0), v_b0);
        HVX_ಸದಿಶ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ಅಡ್ಡಸಾಲುಗಳು_kk[t1], v_d1), v_b1);

        p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
    }

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 32 + 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_ಸದಿಶ v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_ಸದಿಶ v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_ಸದಿಶ v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_ಸದಿಶ v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_ಸದಿಶ r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ಅಡ್ಡಸಾಲುಗಳು_kk[t0], v_d0), v_b0);
        HVX_ಸದಿಶ r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(ಅಡ್ಡಸಾಲುಗಳು_kk[t1], v_d1), v_b1);

        HVX_ಸದಿಶ_Pair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
        p_l10_tile[r] = Q6_V_lo_W(vp_l10);
        p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
    }

    HVX_ಸದಿಶ a_ಅಡ್ಡಸಾಲುಗಳು[32];
    ಗಾಗಿ (uint32_t t = 0; t < 32; ++t) {
        HVX_ಸದಿಶ v_inv = Q6_V_vzero();
        ಗಾಗಿ (uint32_t k = 0; k < t; ++k) {
            HVX_ಸದಿಶ v_lk = Q6_Vh_vsplat_R(l00[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_ಅಡ್ಡಸಾಲುಗಳು[k]));
        }
        HVX_ಸದಿಶ_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_ಅಡ್ಡಸಾಲುಗಳು[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(a_ಅಡ್ಡಸಾಲುಗಳು[2 * r + 1], a_ಅಡ್ಡಸಾಲುಗಳು[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    ಗಾಗಿ (uint32_t t = 0; t < 32; ++t) {
        HVX_ಸದಿಶ v_inv = Q6_V_vzero();
        ಗಾಗಿ (uint32_t k = 0; k < t; ++k) {
            HVX_ಸದಿಶ v_lk = Q6_Vh_vsplat_R(l11[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_ಅಡ್ಡಸಾಲುಗಳು[k]));
        }
        HVX_ಸದಿಶ_Pred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_ಅಡ್ಡಸಾಲುಗಳು[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    ಗಾಗಿ (uint32_t r = 0; r < 16; ++r) {
        HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(a_ಅಡ್ಡಸಾಲುಗಳು[2 * r + 1], a_ಅಡ್ಡಸಾಲುಗಳು[2 * r + 0], -2);
        tile11[r] = Q6_V_lo_W(vp);

        HVX_ಸದಿಶ n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_ಅಡ್ಡಸಾಲುಗಳು[2 * r + 0]);
        HVX_ಸದಿಶ n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_ಅಡ್ಡಸಾಲುಗಳು[2 * r + 1]);
        HVX_ಸದಿಶ_Pair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
        p_neg_a11[r] = Q6_V_lo_W(vp_neg);
    }
}


ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_dma_ತಳ್ಳು_chunk_inputs(
    dma_ಸರತಿ * dma_q,
    ದಶಮಾಂಶ * vtcm_q,
    ದಶಮಾಂಶ * vtcm_k,
    ದಶಮಾಂಶ * vtcm_v,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_ಗಾತ್ರ,
    uint32_t S_v
) {
    ಸ್ಥಿರಾಂಕ dma_addr_t q_dma = q->ದತ್ತಾಂಶ + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
    ಸ್ಥಿರಾಂಕ dma_addr_t k_dma = k->ದತ್ತಾಂಶ + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
    ಸ್ಥಿರಾಂಕ dma_addr_t v_dma = v->ದತ್ತಾಂಶ + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_q, q_dma), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), q->nb[2], S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), chunk_ಗಾತ್ರ);
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_k, k_dma), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), k->nb[2], S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), chunk_ಗಾತ್ರ);
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_v, v_dma), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), v->nb[2], S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), chunk_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_dma_ತಳ್ಳು_chunk_gb(
    dma_ಸರತಿ * dma_q,
    ದಶಮಾಂಶ * vtcm_g_raw,
    ದಶಮಾಂಶ * vtcm_b_raw,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * g,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_ಗಾತ್ರ,
    uint32_t n_batch
) {
    ಸ್ಥಿರಾಂಕ dma_addr_t g_dma    = g->ದತ್ತಾಂಶ + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    ಸ್ಥಿರಾಂಕ dma_addr_t beta_dma = beta->ದತ್ತಾಂಶ + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು  = n_batch * ಗಾತ್ರ(ದಶಮಾಂಶ);

    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_g_raw, g_dma), ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, g->nb[2], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, chunk_ಗಾತ್ರ);
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_b_raw, beta_dma), ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, beta->nb[2], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, chunk_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_pack_s_ಲಂಬಸಾಲು_tiles(
    __fp16 * ಸೀಮಿತ vtcm_s_ಲಂಬಸಾಲು_tiles,
    __fp16 * ಸೀಮಿತ vtcm_s_f16,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ vtcm_s_state,
    uint32_t S_v
) {
    ಗಾಗಿ (uint32_t j = 0; j < S_v; ++j) {
        ಗಾಗಿ (uint32_t i = 0; i < S_v; i += 64) {
            HVX_ಸದಿಶ v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_ಸದಿಶ v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_ಗಮ್ಯ_f16(v0, v1);
        }
    }
    hmx_interleave_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_tiles(vtcm_s_ಲಂಬಸಾಲು_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);
}

ರಚನೆ htp_gdn_ತಲೆ_ptrs {
    ದಶಮಾಂಶ *  s_state;
    __fp16 * s_f16;
    __fp16 * s_ಲಂಬಸಾಲು_tiles;
    ದಶಮಾಂಶ *  s_update_f32;
    __fp16 * s_update_tiles;

    ದಶಮಾಂಶ * q_f32[2];
    ದಶಮಾಂಶ * k_f32[2];
    ದಶಮಾಂಶ * v_f32[2];
    ದಶಮಾಂಶ * g_f32[2];
    ದಶಮಾಂಶ * b_f32[2];
    ದಶಮಾಂಶ * o_f32[2];

    ದಶಮಾಂಶ * v_inter_f32;
    ದಶಮಾಂಶ * o_inter_f32;
    ದಶಮಾಂಶ * o_intra_f32;

    __fp16 * k_f16;
    __fp16 * v_ಅವಿಭಾಜ್ಯ_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_ಅಡ್ಡಸಾಲು_tiles;
    __fp16 * q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles;
    __fp16 * k_ಅಡ್ಡಸಾಲು_tiles;
    __fp16 * k_ಲಂಬಸಾಲು_tiles;
    __fp16 * k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles;
    __fp16 * k_ಲಂಬಸಾಲು_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_ಅಡ್ಡಸಾಲು_tiles;
    __fp16 * a_ಅಡ್ಡಸಾಲು_tiles;
    __fp16 * v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_ಲಂಬಸಾಲು_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_ಅಡ್ಡಸಾಲು_tiles;

    __fp16 * gamma;
    ದಶಮಾಂಶ *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_ಸದಿಶ * ಅಡ್ಡಸಾಲುಗಳು_kk;
    HVX_ಸದಿಶ * ಅಡ್ಡಸಾಲುಗಳು_qk;
    HVX_ಸದಿಶ * ಅಡ್ಡಸಾಲುಗಳು_inv;
    HVX_ಸದಿಶ * ಅಡ್ಡಸಾಲುಗಳು_a;

    HVX_ಸದಿಶ * vtcm_m;
    HVX_ಸದಿಶ * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_ಒಳ_dma;
    dma_addr_t state_ಹೊರಹೋಗು_dma;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gdn_init_ತಲೆ_ptrs(
    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_ತಳ,
    uint32_t h,
    uint32_t ತಳ_iv1,
    uint32_t iv3,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * state,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst_cache,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_ಟೋಕನುಗಳು,
    uint32_t chunk_ಗಾತ್ರ
) {
    ಸ್ಥಿರಾಂಕ size_t dma_scalar_sz = hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
    ಸ್ಥಿರಾಂಕ size_t decay_sz      = 64 * 64 * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_vecs_sz   = 64 * 128;

    ತಲೆ->s_state        = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_s_state + h * L->state_f32_ಬೈಟುಗಳು);
    ತಲೆ->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_s_f16 + h * L->state_f16_ಬೈಟುಗಳು);
    ತಲೆ->s_ಲಂಬಸಾಲು_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_s_ಲಂಬಸಾಲು_tiles + h * L->state_tiles_ಬೈಟುಗಳು);
    ತಲೆ->s_update_f32   = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_s_update_f32 + h * L->state_f32_ಬೈಟುಗಳು);
    ತಲೆ->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_s_update_tiles + h * L->state_tiles_ಬೈಟುಗಳು);

    ತಲೆ->q_f32[0] = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_q_f32[0] + h * L->dma_chunk_ಬೈಟುಗಳು);
    ತಲೆ->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_q_f32[1] + h * L->dma_chunk_ಬೈಟುಗಳು) : ತಲೆ->q_f32[0];
    ತಲೆ->k_f32[0] = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_k_f32[0] + h * L->dma_chunk_ಬೈಟುಗಳು);
    ತಲೆ->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_k_f32[1] + h * L->dma_chunk_ಬೈಟುಗಳು) : ತಲೆ->k_f32[0];
    ತಲೆ->v_f32[0] = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_v_f32[0] + h * L->dma_chunk_ಬೈಟುಗಳು);
    ತಲೆ->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_v_f32[1] + h * L->dma_chunk_ಬೈಟುಗಳು) : ತಲೆ->v_f32[0];
    ತಲೆ->g_f32[0] = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_g_f32[0] + h * dma_scalar_sz);
    ತಲೆ->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_g_f32[1] + h * dma_scalar_sz) : ತಲೆ->g_f32[0];
    ತಲೆ->b_f32[0] = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_b_f32[0] + h * dma_scalar_sz);
    ತಲೆ->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_b_f32[1] + h * dma_scalar_sz) : ತಲೆ->b_f32[0];
    ತಲೆ->o_f32[0] = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_o_f32[0] + h * L->dma_chunk_ಬೈಟುಗಳು);
    ತಲೆ->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_o_f32[1] + h * L->dma_chunk_ಬೈಟುಗಳು) : ತಲೆ->o_f32[0];

    ತಲೆ->v_inter_f32 = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_v_inter_f32 + h * L->dma_chunk_ಬೈಟುಗಳು);
    ತಲೆ->o_inter_f32 = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_o_inter_f32 + h * L->dma_chunk_ಬೈಟುಗಳು);
    ತಲೆ->o_intra_f32 = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_o_intra_f32 + h * L->dma_chunk_ಬೈಟುಗಳು);

    ತಲೆ->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_k_f16 + h * L->act_f16_ಬೈಟುಗಳು);
    ತಲೆ->v_ಅವಿಭಾಜ್ಯ_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_v_ಅವಿಭಾಜ್ಯ_f16 + h * L->act_f16_ಬೈಟುಗಳು);
    ತಲೆ->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_delta_f16 + h * L->act_f16_ಬೈಟುಗಳು);
    ತಲೆ->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_d_f16 + h * L->act_f16_ಬೈಟುಗಳು);

    ತಲೆ->q_ಅಡ್ಡಸಾಲು_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_q_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->k_ಅಡ್ಡಸಾಲು_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_k_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->k_ಲಂಬಸಾಲು_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_k_ಲಂಬಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->k_ಲಂಬಸಾಲು_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_k_ಲಂಬಸಾಲು_tiles_64x128 + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_kk_tiles + h * L->tile_64x64_ಬೈಟುಗಳು);
    ತಲೆ->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_qk_tiles + h * L->tile_64x64_ಬೈಟುಗಳು);
    ತಲೆ->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_v_inter_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_o_inter_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->inv_ಅಡ್ಡಸಾಲು_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_inv_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x64_ಬೈಟುಗಳು);
    ತಲೆ->a_ಅಡ್ಡಸಾಲು_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_a_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x64_ಬೈಟುಗಳು);
    ತಲೆ->v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_delta_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->delta_ಲಂಬಸಾಲು_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_delta_ಲಂಬಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_o_intra_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);
    ತಲೆ->d_ಅಡ್ಡಸಾಲು_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_d_ಅಡ್ಡಸಾಲು_tiles + h * L->tile_64x_Sv_ಬೈಟುಗಳು);

    ತಲೆ->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_gamma + h * dma_scalar_sz);
    ತಲೆ->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_gamma + h * dma_scalar_sz + 128);
    ತಲೆ->lambda_init     = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L->off_lambda_init + h * dma_scalar_sz);
    ತಲೆ->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_decay_m + h * decay_sz);
    ತಲೆ->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_ತಳ, L->off_decay_a + h * decay_sz);

    ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_kk  = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, vtcm_ತಳ, L->off_ಅಡ್ಡಸಾಲುಗಳು_kk + h * ಅಡ್ಡಸಾಲು_vecs_sz);
    ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_qk  = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, vtcm_ತಳ, L->off_ಅಡ್ಡಸಾಲುಗಳು_qk + h * ಅಡ್ಡಸಾಲು_vecs_sz);
    ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_inv = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, vtcm_ತಳ, L->off_ಅಡ್ಡಸಾಲುಗಳು_inv + h * ಅಡ್ಡಸಾಲು_vecs_sz);
    ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_a   = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, vtcm_ತಳ, L->off_ಅಡ್ಡಸಾಲುಗಳು_a + h * ಅಡ್ಡಸಾಲು_vecs_sz);

    ತಲೆ->vtcm_m   = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, vtcm_ತಳ, L->off_thread_scratch + h * (64 * 128));
    ತಲೆ->vtcm_tmp = ತಲೆ->vtcm_m + 32;

    ತಲೆ->iv1 = ತಳ_iv1 + h;
    ತಲೆ->iv3 = iv3;
    ತಲೆ->iq1 = fastmodulo(ತಲೆ->iv1, q->ne[1], &kparams->div_q1);
    ತಲೆ->ik1 = fastmodulo(ತಲೆ->iv1, k->ne[1], &kparams->div_k1);
    ತಲೆ->iq3 = fastdiv(ತಲೆ->iv3, &kparams->div_rq3);
    ತಲೆ->ik3 = fastdiv(ತಲೆ->iv3, &kparams->div_rk3);

    ತಲೆ->state_ಒಳ_dma = state->ದತ್ತಾಂಶ +
        ((uint64_t) ತಲೆ->iv3 * kparams->state_seq_stride + (uint64_t) ತಲೆ->iv1 * S_v * S_v) * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ತಲೆ->state_ಹೊರಹೋಗು_dma = dst_cache ?
        (dst_cache->ದತ್ತಾಂಶ + ((uint64_t) ತಲೆ->iv3 * H + ತಲೆ->iv1) * S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ)) :
        (dst->ದತ್ತಾಂಶ + ((uint64_t) S_v * H * n_ಟೋಕನುಗಳು * kparams->n_seqs + (uint64_t) (ತಲೆ->iv3 * H + ತಲೆ->iv1) * S_v * S_v) * ಗಾತ್ರ(ದಶಮಾಂಶ));
}

ರಚನೆ htp_gdn_batch_context {
    ರಚನೆ htp_gdn_ತಲೆ_ptrs * heads;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *              vtcm_g_raw;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    ದಶಮಾಂಶ                      scale;
    ರಚನೆ htp_ops_context *   octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams;
};

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_init_state_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    gdn_pack_s_ಲಂಬಸಾಲು_tiles(ತಲೆ->s_ಲಂಬಸಾಲು_tiles, ತಲೆ->s_f16, ತಲೆ->s_state, bctx->S_v);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ __attribute__((ಬಳಸದ_2)) HVX_ಸದಿಶ hvx_clamp_neg20_0(HVX_ಸದಿಶ v, HVX_ಸದಿಶ v_zero, HVX_ಸದಿಶ v_neg20) {
    HVX_ಸದಿಶ_Pred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_ಸದಿಶ_Pred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_ಪೂರ್ವಪ್ರತ್ಯಯ_ಪರಿಶೀಲನೆ_f32(HVX_ಸದಿಶ v, HVX_ಸದಿಶ ವಾಹಕ_ಒಳ) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = Q6_V_vzero();

    v = hvx_vec_ಕೂಡಿಸು_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_ಕೂಡಿಸು_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_ಕೂಡಿಸು_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_ಕೂಡಿಸು_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_ಕೂಡಿಸು_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_ಕೂಡಿಸು_f32_f32(v, ವಾಹಕ_ಒಳ);

    ಹಿಂತಿರುಗಿ v;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_splat_ಕೊನೆಯ_f32(HVX_ಸದಿಶ v) {
    ಹಿಂತಿರುಗಿ hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase1a_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t curr_buf = bctx->curr_buf;
    ಸ್ಥಿರಾಂಕ uint32_t S_v = bctx->S_v;
    ಸ್ಥಿರಾಂಕ uint32_t n_batch = bctx->n_batch;

    ಆದರೆ (n_batch == 1) {
        hvx_vmem(ತಲೆ->g_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(ತಲೆ->g_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(ತಲೆ->b_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(ತಲೆ->b_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } ಇಲ್ಲದಿದ್ದರೆ {
        int32_t offsets[32] __attribute__((aligned(128)));
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 32; ++k) {
            offsets[k] = k * n_batch * ಗಾತ್ರ(ದಶಮಾಂಶ);
        }
        HVX_ಸದಿಶ vv = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) offsets;
        ಸ್ಥಿರಾಂಕ size_t rt_g = (size_t) ((ಸ್ಥಿರಾಂಕ uint8_t *) bctx->vtcm_g_raw + i * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ size_t rt_b = (size_t) ((ಸ್ಥಿರಾಂಕ uint8_t *) bctx->vtcm_b_raw + i * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ size_t mu   = 64 * n_batch * ಗಾತ್ರ(ದಶಮಾಂಶ);

        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (ತಲೆ->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (ತಲೆ->g_f32[curr_buf] + 32), rt_g + 32 * n_batch * ಗಾತ್ರ(ದಶಮಾಂಶ), mu, vv);
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (ತಲೆ->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (ತಲೆ->b_f32[curr_buf] + 32), rt_b + 32 * n_batch * ಗಾತ್ರ(ದಶಮಾಂಶ), mu, vv);
    }

    ಸ್ಥಿರಾಂಕ uint32_t t_chunk = bctx->c * 64;
    ಸ್ಥಿರಾಂಕ uint32_t valid_ಟೋಕನುಗಳು = hex_smin(64, bctx->kparams->n_ಟೋಕನುಗಳು - t_chunk);
    ಆದರೆ (valid_ಟೋಕನುಗಳು < 64) {
        ಗಾಗಿ (uint32_t t = valid_ಟೋಕನುಗಳು; t < 64; ++t) {
            ತಲೆ->g_f32[curr_buf][t] = 0.0f;
            ತಲೆ->b_f32[curr_buf][t] = 0.0f;
        }
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vzero = Q6_V_vzero();
        ಗಾಗಿ (uint32_t t = valid_ಟೋಕನುಗಳು; t < 64; ++t) {
            ಗಾಗಿ (uint32_t j = 0; j < S_v; j += 32) {
                hvx_vmem(ತಲೆ->q_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(ತಲೆ->k_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(ತಲೆ->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_g0 = hvx_vmem(ತಲೆ->g_f32[curr_buf] + 0);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_g1 = hvx_vmem(ತಲೆ->g_f32[curr_buf] + 32);

    HVX_ಸದಿಶ v_gamma0 = hvx_ಪೂರ್ವಪ್ರತ್ಯಯ_ಪರಿಶೀಲನೆ_f32(v_g0, Q6_V_vzero());
    HVX_ಸದಿಶ v_ವಾಹಕ  = hvx_splat_ಕೊನೆಯ_f32(v_gamma0);
    HVX_ಸದಿಶ v_gamma1 = hvx_ಪೂರ್ವಪ್ರತ್ಯಯ_ಪರಿಶೀಲನೆ_f32(v_g1, v_ವಾಹಕ);

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_zero  = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg20 = hvx_vec_splat_f32(-20.0f);

    hvx_vmem(ತಲೆ->gamma) = hvx_vec_f32_ಗಮ್ಯ_f16(v_gamma0, v_gamma1);

    HVX_ಸದಿಶ v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_ಸದಿಶ v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    hvx_vmem(ತಲೆ->lambda_init + 0)  = v_l0;
    hvx_vmem(ತಲೆ->lambda_init + 32) = v_l1;
    hvx_vmem(ತಲೆ->lambda_init_f16)  = hvx_vec_f32_ಗಮ್ಯ_f16(v_l0, v_l1);

    gdn_f32_ಗಮ್ಯ_hmx_ಅಡ್ಡಸಾಲು_tiles_and_f16(ತಲೆ->k_ಅಡ್ಡಸಾಲು_tiles, ತಲೆ->k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles, ತಲೆ->k_f16,
                                     ತಲೆ->k_f32[curr_buf], ತಲೆ->lambda_init_f16, 64, S_v);
    hmx_interleave_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_tiles(ತಲೆ->k_ಲಂಬಸಾಲು_tiles, ತಲೆ->k_f16, 64, S_v, S_v, 0, 64);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase1b_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t curr_buf = bctx->curr_buf;
    ಸ್ಥಿರಾಂಕ uint32_t S_v = bctx->S_v;

    gdn_f32_ಗಮ್ಯ_hmx_ಅಡ್ಡಸಾಲು_tiles_and_f16(ತಲೆ->q_ಅಡ್ಡಸಾಲು_tiles, ತಲೆ->q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles, ಖಾಲಿ,
                                     ತಲೆ->q_f32[curr_buf], ತಲೆ->lambda_init_f16, 64, S_v);

    hmx_interleave_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಗಮ್ಯ_tiles(ತಲೆ->k_ಲಂಬಸಾಲು_tiles_64x128, ತಲೆ->k_f16, 64, S_v, S_v, 2, 0, 64);

    ಸ್ಥಿರಾಂಕ uint16_t * gamma_u16 = (ಸ್ಥಿರಾಂಕ uint16_t *) ತಲೆ->gamma;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_gamma   = hvx_vmem(ತಲೆ->gamma);

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_zero_f16  = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(ತಲೆ->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(ತಲೆ->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    ಗಾಗಿ (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_ಸದಿಶ v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_ಸದಿಶ v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_ಸದಿಶ diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_ಸದಿಶ diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_ಸದಿಶ_Pred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_ಸದಿಶ_Pred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_ಸದಿಶ diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_ಸದಿಶ diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_ಸದಿಶ v_exp0      = hvx_vec_exp2_f16(diff_log2e0);
        HVX_ಸದಿಶ v_exp1      = hvx_vec_exp2_f16(diff_log2e1);

        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_ಸದಿಶ v_m0         = Q6_V_vand_QV(ಮುಖವಾಡ_lt0, v_exp0);
        HVX_ಸದಿಶ v_m1         = Q6_V_vand_QV(ಮುಖವಾಡ_lt1, v_exp1);

        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_diag0 = Q6_Q_and_QQn(ಮುಖವಾಡ_le0, ಮುಖವಾಡ_lt0);
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_diag1 = Q6_Q_and_QQn(ಮುಖವಾಡ_le1, ಮುಖವಾಡ_lt1);

        HVX_ಸದಿಶ v_a0         = Q6_V_vmux_QVV(ಮುಖವಾಡ_diag0, v_one_f16, v_m0);
        HVX_ಸದಿಶ v_a1         = Q6_V_vmux_QVV(ಮುಖವಾಡ_diag1, v_one_f16, v_m1);

        hvx_vmem(ತಲೆ->decay_m + t0 * 64) = v_m0;
        hvx_vmem(ತಲೆ->decay_a + t0 * 64) = v_a0;
        hvx_vmem(ತಲೆ->decay_m + t1 * 64) = v_m1;
        hvx_vmem(ತಲೆ->decay_a + t1 * 64) = v_a1;
    }

    {
        HVX_ಸದಿಶ v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_ಸದಿಶ diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_ಸದಿಶ_Pred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_ಸದಿಶ diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_ಸದಿಶ v_exp         = hvx_vec_exp2_f16(diff_log2e);

        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_ಸದಿಶ v_m           = Q6_V_vand_QV(ಮುಖವಾಡ_lt_t, v_exp);

        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_diag = Q6_Q_and_QQn(ಮುಖವಾಡ_le_t, ಮುಖವಾಡ_lt_t);
        HVX_ಸದಿಶ v_a           = Q6_V_vmux_QVV(ಮುಖವಾಡ_diag, v_one_f16, v_m);

        hvx_vmem(ತಲೆ->decay_m + 63 * 64) = v_m;
        hvx_vmem(ತಲೆ->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase2_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t curr_buf = bctx->curr_buf;

    gdn_unpack_64x64_tiles_ಗಮ್ಯ_vectors(ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_kk, ತಲೆ->kk_tiles);

    gdn_build_inv_l_ಖಂಡಗಳು(
        ತಲೆ->inv_ಅಡ್ಡಸಾಲು_tiles,
        ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_kk,
        ತಲೆ->decay_m,
        ತಲೆ->b_f32[curr_buf],
        (__fp16 *) ತಲೆ->vtcm_m,
        (__fp16 *) ತಲೆ->vtcm_m + HMX_FP16_TILE_N_ELMS
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase3_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t curr_buf = bctx->curr_buf;
    ಸ್ಥಿರಾಂಕ uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_ಗಮ್ಯ_f32(ತಲೆ->v_inter_f32, ತಲೆ->v_inter_tiles, S_v);

    HVX_ಸದಿಶ_Alias local_b[2];
    local_b[0].v = hvx_vmem(ತಲೆ->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(ತಲೆ->b_f32[curr_buf] + 32);

    ಗಾಗಿ (uint32_t t = 0; t < 64; ++t) {
        HVX_ಸದಿಶ vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
        ಗಾಗಿ (uint32_t j = 0; j < S_v; j += 64) {
            HVX_ಸದಿಶ vv0 = hvx_vmem(ತಲೆ->v_f32[curr_buf] + t * S_v + j + 0);
            HVX_ಸದಿಶ vv1 = (j + 32 < S_v) ? hvx_vmem(ತಲೆ->v_f32[curr_buf] + t * S_v + j + 32) : Q6_V_vzero();
            HVX_ಸದಿಶ vi0 = hvx_vmem(ತಲೆ->v_inter_f32 + t * S_v + j + 0);
            HVX_ಸದಿಶ vi1 = (j + 32 < S_v) ? hvx_vmem(ತಲೆ->v_inter_f32 + t * S_v + j + 32) : Q6_V_vzero();

            HVX_ಸದಿಶ vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
            HVX_ಸದಿಶ vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

            hvx_vmem(ತಲೆ->v_ಅವಿಭಾಜ್ಯ_f16 + t * S_v + j) = hvx_vec_f32_ಗಮ್ಯ_f16(vp0, vp1);
        }
    }

    hmx_interleave_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಗಮ್ಯ_tiles(ತಲೆ->v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles, ತಲೆ->v_ಅವಿಭಾಜ್ಯ_f16, 64, S_v, S_v, 2, 0, 64);

    gdn_unpack_64x64_tiles_ಗಮ್ಯ_vectors(ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_qk, ತಲೆ->qk_tiles);
    ಗಾಗಿ (uint32_t t = 0; t < 64; ++t) {
        HVX_ಸದಿಶ v_decay_a = hvx_vmem(ತಲೆ->decay_a + t * 64);
        ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_a[t]      = hvx_vec_mul_f16_f16(ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_ಗಮ್ಯ_tiles(ತಲೆ->a_ಅಡ್ಡಸಾಲು_tiles, ತಲೆ->ಅಡ್ಡಸಾಲುಗಳು_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase4_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t S_v = bctx->S_v;

    gdn_unpack_64x_S_tiles_ಗಮ್ಯ_f16(ತಲೆ->delta_f16, ತಲೆ->delta_tiles, S_v);
    hmx_interleave_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಗಮ್ಯ_tiles(ತಲೆ->delta_ಲಂಬಸಾಲು_tiles, ತಲೆ->delta_f16, 64, S_v, S_v, 2, 0, 64);

    ಸ್ಥಿರಾಂಕ uint16_t * decay_ಕೊನೆಯ = (ಸ್ಥಿರಾಂಕ uint16_t *) (ತಲೆ->decay_a + 63 * 64);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vzero      = Q6_V_vzero();

    ಗಾಗಿ (uint32_t s = 0; s < 64; ++s) {
        HVX_ಸದಿಶ vs         = Q6_Vh_vsplat_R(decay_ಕೊನೆಯ[s]);
        HVX_ಸದಿಶ_Pred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
        ಗಾಗಿ (uint32_t j = 0; j < S_v; j += 64) {
            HVX_ಸದಿಶ vd   = hvx_vmem(ತಲೆ->delta_f16 + s * S_v + j);
            HVX_ಸದಿಶ prod = hvx_vec_mul_f16_f16(vd, vs);
            hvx_vmem(ತಲೆ->d_f16 + s * S_v + j) = Q6_V_vmux_QVV(p_zero, vzero, prod);
        }
    }

    gdn_pack_d_t_ಅಡ್ಡಸಾಲು_tiles(ತಲೆ->d_ಅಡ್ಡಸಾಲು_tiles, ತಲೆ->d_f16, S_v, ತಲೆ->vtcm_m, ತಲೆ->vtcm_tmp);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase5_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_ಹೊರಹೋಗು, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t curr_buf = bctx->curr_buf;
    ಸ್ಥಿರಾಂಕ uint32_t S_v = bctx->S_v;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = bctx->scale;

    gdn_unpack_64x_S_tiles_ಗಮ್ಯ_f32(ತಲೆ->o_inter_f32, ತಲೆ->o_inter_tiles, S_v);
    gdn_unpack_64x_S_tiles_ಗಮ್ಯ_f32(ತಲೆ->o_intra_f32, ತಲೆ->o_intra_tiles, S_v);

    HVX_ಸದಿಶ vscale = hvx_vec_splat_f32(scale);
    ಗಾಗಿ (uint32_t j = 0; j < 64 * S_v / 32; ++j) {
        HVX_ಸದಿಶ vi = hvx_vmem(ತಲೆ->o_inter_f32 + j * 32);
        HVX_ಸದಿಶ va = hvx_vmem(ತಲೆ->o_intra_f32 + j * 32);
        hvx_vmem(ತಲೆ->o_f32[curr_buf] + j * 32) = hvx_vec_mul_f32_f32(hvx_vec_ಕೂಡಿಸು_f32_f32(vi, va), vscale);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_ಹೊರಹೋಗು, info);
}

ಸ್ಥಿರ ಶೂನ್ಯ gdn_hvx_phase6_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    ರಚನೆ htp_gdn_batch_context * bctx = (ರಚನೆ htp_gdn_batch_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    ರಚನೆ htp_gdn_ತಲೆ_ptrs * ತಲೆ = &bctx->heads[i];
    ಸ್ಥಿರಾಂಕ uint32_t S_v = bctx->S_v;
    ಸ್ಥಿರಾಂಕ uint32_t c = bctx->c;
    ಸ್ಥಿರಾಂಕ uint32_t n_chunks = bctx->kparams->n_chunks;

    gdn_unpack_Sx_S_tiles_ಗಮ್ಯ_f32(ತಲೆ->s_update_f32, ತಲೆ->s_update_tiles, S_v);

    HVX_ಸದಿಶ_Alias ಕೊನೆಯ_lambda;
    ಕೊನೆಯ_lambda.v = hvx_vmem(ತಲೆ->lambda_init + 32);
    HVX_ಸದಿಶ v_l_final = hvx_vec_splat_f32(ಕೊನೆಯ_lambda.fp32[31]);

    ಗಾಗಿ (uint32_t j = 0; j < S_v * S_v / 32; ++j) {
        HVX_ಸದಿಶ vs_old = hvx_vmem(ತಲೆ->s_state + j * 32);
        HVX_ಸದಿಶ vsu    = hvx_vmem(ತಲೆ->s_update_f32 + j * 32);
        hvx_vmem(ತಲೆ->s_state + j * 32) = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
    }

    ಆದರೆ (c + 1 < n_chunks) {
        gdn_pack_s_ಲಂಬಸಾಲು_tiles(ತಲೆ->s_ಲಂಬಸಾಲು_tiles, ತಲೆ->s_f16, ತಲೆ->s_state, S_v);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ gated_delta_net_f32_hmx_chunked(
    ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams,
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ,
    uint32_t nrows
) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q         = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k         = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v         = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * g         = octx->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * beta      = octx->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * state     = octx->src[5];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst       = octx->dst;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst_cache = octx->dsts[1];

    ಸ್ಥಿರಾಂಕ uint32_t S_v        = kparams->S_v;
    ಸ್ಥಿರಾಂಕ uint32_t H          = kparams->H;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು   = kparams->n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    scale      = kparams->scale;
    ಸ್ಥಿರಾಂಕ uint32_t chunk_ಗಾತ್ರ = kparams->chunk_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t n_chunks   = kparams->n_chunks;
    ಸ್ಥಿರಾಂಕ uint32_t n_sv_tiles = S_v / 32;

    ರಚನೆ htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_ಗಾತ್ರ, kparams->n_heads_batch, kparams->n_ಎಳೆಗಳು, kparams->pipeline != 0);

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ vtcm_ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;

    ದಶಮಾಂಶ * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L.off_g_raw[0])
    };
    ದಶಮಾಂಶ * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, vtcm_ತಳ, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_ತಳ, L.off_scales_1);
    hmx_init_ಸ್ತಂಭ_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_ಸರತಿ_t  hmx_q = octx->ctx->hmx_ಸರತಿ;
    dma_ಸರತಿ *  dma_q = octx->ctx->dma[0];
    work_ಸರತಿ_t wp    = octx->ctx->work_ಸರತಿ;

    ರಚನೆ htp_gdn_ತಲೆ_ptrs heads[8];
    ರಚನೆ htp_gdn_hmx_gemm_task gemm_tasks[8][9];

    uint32_t n_batch = 1;
    ಗಾಗಿ (uint32_t r = ಅಡ್ಡಸಾಲು_ಆರಂಭ; r < ಅಡ್ಡಸಾಲು_ಆರಂಭ + nrows; r += n_batch) {
        ಸ್ಥಿರಾಂಕ uint32_t ತಲೆ_ಒಳ_seq         = fastmodulo(r, H, &kparams->div_H);
        ಸ್ಥಿರಾಂಕ uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        ಸ್ಥಿರಾಂಕ uint32_t heads_ಎಡ_ಒಳ_seq   = H - ತಲೆ_ಒಳ_seq;
        ಸ್ಥಿರಾಂಕ uint32_t heads_ಎಡ_ಒಳ_range = (ಅಡ್ಡಸಾಲು_ಆರಂಭ + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_ಎಡ_ಒಳ_seq, heads_ಎಡ_ಒಳ_range));

        ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_ತಲೆ_ptrs(&heads[h], &L, vtcm_ತಳ, h, ತಲೆ_ಒಳ_seq, iv3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_ಟೋಕನುಗಳು, chunk_ಗಾತ್ರ);
        }

        ರಚನೆ htp_gdn_batch_context bctx;
        bctx.heads      = heads;
        bctx.vtcm_g_raw = ಖಾಲಿ;
        bctx.vtcm_b_raw = ಖಾಲಿ;
        bctx.curr_buf   = 0;
        bctx.c          = 0;
        bctx.n_batch    = n_batch;
        bctx.S_v        = S_v;
        bctx.scale      = scale;
        bctx.octx       = octx;
        bctx.kparams    = kparams;

        ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(heads[h].s_state, heads[h].state_ಒಳ_dma),
                           S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);
        }
        ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }

        ಆದರೆ (n_chunks > 0) {
            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_init_state_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಸ್ಥಿರಾಂಕ uint32_t chunk0_ಟೋಕನುಗಳು = hex_smin(chunk_ಗಾತ್ರ, n_ಟೋಕನುಗಳು);
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                gdn_dma_ತಳ್ಳು_chunk_inputs(dma_q, heads[h].q_f32[0], heads[h].k_f32[0], heads[h].v_f32[0],
                                          q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                          heads[h].iv3, heads[h].iv1, 0, chunk0_ಟೋಕನುಗಳು, S_v);
            }
            gdn_dma_ತಳ್ಳು_chunk_gb(dma_q, vtcm_g_raw[0], vtcm_b_raw[0], g, beta, iv3, ತಲೆ_ಒಳ_seq, 0, chunk0_ಟೋಕನುಗಳು, n_batch);
        }

        ಗಾಗಿ (uint32_t c = 0; c < n_chunks; ++c) {
            ಸ್ಥಿರಾಂಕ uint32_t curr_buf = c & 1;
            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_buf = (c + 1) & 1;
            ಸ್ಥಿರಾಂಕ uint32_t t_chunk  = c * chunk_ಗಾತ್ರ;

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                dma_ಸರತಿ_ತೆಗೆ(dma_q);
                dma_ಸರತಿ_ತೆಗೆ(dma_q);
                dma_ಸರತಿ_ತೆಗೆ(dma_q);
            }
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
            dma_ಸರತಿ_ತೆಗೆ(dma_q);

            ಆದರೆ (c + 1 < n_chunks) {
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_t_chunk = (c + 1) * chunk_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಟೋಕನುಗಳು_2  = hex_smin(chunk_ಗಾತ್ರ, n_ಟೋಕನುಗಳು - ಮುಂದಿನ_t_chunk);
                ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_ತಳ್ಳು_chunk_inputs(dma_q, heads[h].q_f32[ಮುಂದಿನ_buf], heads[h].k_f32[ಮುಂದಿನ_buf], heads[h].v_f32[ಮುಂದಿನ_buf],
                                              q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                              heads[h].iv3, heads[h].iv1, ಮುಂದಿನ_t_chunk, ಮುಂದಿನ_ಟೋಕನುಗಳು_2, S_v);
                }
                gdn_dma_ತಳ್ಳು_chunk_gb(dma_q, vtcm_g_raw[ಮುಂದಿನ_buf], vtcm_b_raw[ಮುಂದಿನ_buf],
                                      g, beta, iv3, ತಲೆ_ಒಳ_seq, ಮುಂದಿನ_t_chunk, ಮುಂದಿನ_ಟೋಕನುಗಳು_2, n_batch);
            }

            ಆದರೆ (c > 0) {
                ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                    dma_ಸರತಿ_ತೆಗೆ(dma_q);
                }
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase1a_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][0], heads[h].k_ಅಡ್ಡಸಾಲು_tiles, heads[h].k_ಲಂಬಸಾಲು_tiles, heads[h].kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][2], heads[h].k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles, heads[h].s_ಲಂಬಸಾಲು_tiles, heads[h].v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase1b_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][1], heads[h].q_ಅಡ್ಡಸಾಲು_tiles, heads[h].k_ಲಂಬಸಾಲು_tiles, heads[h].qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][3], heads[h].q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles, heads[h].s_ಲಂಬಸಾಲು_tiles, heads[h].o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase2_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][7],
                    (__fp16 *) heads[h].vtcm_m,
                    heads[h].inv_ಅಡ್ಡಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    1, 1, 1, vtcm_scales_1
                );
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][8],
                    (__fp16 *) heads[h].vtcm_m + HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    heads[h].inv_ಅಡ್ಡಸಾಲು_tiles + 2 * HMX_FP16_TILE_N_ELMS,
                    1, 1, 1, vtcm_scales_1
                );
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase3_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][4], heads[h].inv_ಅಡ್ಡಸಾಲು_tiles, heads[h].v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles, heads[h].delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase4_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][5], heads[h].a_ಅಡ್ಡಸಾಲು_tiles, heads[h].delta_ಲಂಬಸಾಲು_tiles, heads[h].o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_ತಳ್ಳು_hmx_gemm_task(hmx_q, &gemm_tasks[h][6], heads[h].d_ಅಡ್ಡಸಾಲು_tiles, heads[h].k_ಲಂಬಸಾಲು_tiles_64x128, heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);
            }

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase5_ಕೆಲಸಗಾರ, &bctx, n_batch);

            ಸ್ಥಿರಾಂಕ uint32_t valid_ಟೋಕನುಗಳು = hex_smin(chunk_ಗಾತ್ರ, n_ಟೋಕನುಗಳು - t_chunk);
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                ಸ್ಥಿರಾಂಕ dma_addr_t attn_chunk_dma = dst->ದತ್ತಾಂಶ +
                    ((uint64_t) heads[h].iv3 * n_ಟೋಕನುಗಳು * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ);
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), valid_ಟೋಕನುಗಳು);
            }

            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
            }

            work_ಸರತಿ_ಸತತ(wp, gdn_hvx_phase6_ಕೆಲಸಗಾರ, &bctx, n_batch);
        }

        ಆದರೆ (n_chunks > 0) {
            ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
                dma_ಸರತಿ_ತೆಗೆ(dma_q);
            }
        }

        ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(heads[h].state_ಹೊರಹೋಗು_dma, heads[h].s_state),
                           S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), S_v);
        }
        ಗಾಗಿ (uint32_t h = 0; h < n_batch; ++h) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }
    }

    dma_ಸರತಿ_flush(dma_q);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_gated_delta_net(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q     = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k     = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v     = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * g     = octx->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * beta  = octx->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * state = octx->src[5];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst   = octx->dst;

    ಆದರೆ (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t S_v      = v->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t H        = v->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = v->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_seqs   = v->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t K        = octx->op_params[0];

    ಆದರೆ (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_ಟೋಕನುಗಳು == 0 || n_seqs == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    ಆದರೆ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    ಆದರೆ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_ಟೋಕನುಗಳು || k->ne[2] != n_ಟೋಕನುಗಳು || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    ಆದರೆ (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    ಆದರೆ (dst->ne[0] != S_v * H || dst->ne[1] != n_ಟೋಕನುಗಳು * n_seqs + S_v * n_seqs * K) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 5; i++) {
        ಆದರೆ (htp_tensor_is_extended(octx->src[i])) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }
    ಆದರೆ (htp_tensor_is_extended(octx->dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    ಆದರೆ (octx->dsts[1]) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst_cache = octx->dsts[1];
        ಆದರೆ (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params *) octx->kernel_params;
    ರಚನೆ htp_gdn_kernel_params kparams_local;
    ಆದರೆ (!kparams || kparams->S_v == 0) {
        ಸ್ಥಿರಾಂಕ uint32_t rq3 = n_seqs / q->ne[3];
        ಸ್ಥಿರಾಂಕ uint32_t rk3 = n_seqs / k->ne[3];
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = H * n_seqs;
        uint32_t n_ಎಳೆಗಳು = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು < octx->n_ಎಳೆಗಳು) ? ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು : octx->n_ಎಳೆಗಳು;
        ಆದರೆ (n_ಎಳೆಗಳು == 0) {
            n_ಎಳೆಗಳು = 1;
        }

        memset(&kparams_local, 0, ಗಾತ್ರ(kparams_local));
        kparams_local.n_ಎಳೆಗಳು           = n_ಎಳೆಗಳು;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_ಟೋಕನುಗಳು            = n_ಟೋಕನುಗಳು;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು          = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        kparams_local.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread     = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_ಟೋಕನುಗಳು >= HTP_GDN_MIN_ಟೋಕನುಗಳು) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        ರಚನೆ htp_gdn_hmx_vtcm_layout hmx_layout_local;
        ರಚನೆ htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        ಆದರೆ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_ಗಾತ್ರ, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, octx->ctx->vtcm_ಗಾತ್ರ, n_ಎಳೆಗಳು, true, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_ಗಾತ್ರ      = HTP_GDN_CHUNK_ಗಾತ್ರ;
            kparams_local.n_chunks        = (n_ಟೋಕನುಗಳು + HTP_GDN_CHUNK_ಗಾತ್ರ - 1) / HTP_GDN_CHUNK_ಗಾತ್ರ;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_ಗಾತ್ರ       = (uint32_t) hmx_layout_local.ಒಟ್ಟು_ಬೈಟುಗಳು;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_ಬೈಟುಗಳು;
            kparams_local.vtcm_ಪ್ರತಿ_thread = (uint32_t) (hmx_layout_local.ಒಟ್ಟು_ಬೈಟುಗಳು / (n_ಎಳೆಗಳು > 0 ? n_ಎಳೆಗಳು : 1));
        } ಇಲ್ಲದಿದ್ದರೆ {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_ಎಳೆಗಳು);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_ಪ್ರತಿ_thread = (uint32_t) hvx_layout_local.ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
            kparams_local.vtcm_ಗಾತ್ರ       = (uint32_t) hvx_layout_local.ಒಟ್ಟು_ಬೈಟುಗಳು;
        }
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((ದಶಮಾಂಶ) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));
        kparams_local.state_ಗಾತ್ರ_ಪ್ರತಿ_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_ಮೌಲ್ಯಗಳು(H);
        kparams_local.div_q1        = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_ಮೌಲ್ಯಗಳು(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_ಮೌಲ್ಯಗಳು(rq3);
        kparams_local.div_rk3       = init_fastdiv_ಮೌಲ್ಯಗಳು(rk3);
        kparams_local.div_n_ಎಳೆಗಳು = init_fastdiv_ಮೌಲ್ಯಗಳು(n_ಎಳೆಗಳು);

        kparams = &kparams_local;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು,
            can_ವಿಭಜಿಸು ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.ಎಣಿಕೆ_2,
            &octx->ctx->mdev.ಎಣಿಕೆ_div
        );
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (octx->op_params[1] != 0) {
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        ಹಿಂತಿರುಗಿ gated_delta_net_f32_hmx_chunked(octx, kparams, ಅಡ್ಡಸಾಲು_ಆರಂಭ, nrows);
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (nrows < kparams->n_ಎಳೆಗಳು) ? nrows : kparams->n_ಎಳೆಗಳು;

    ರಚನೆ htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.ಅಡ್ಡಸಾಲು_ಆರಂಭ = ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    gctx.nrows     = nrows;
    gctx.vtcm_ತಳ = octx->ctx->vtcm_ತಳ;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_ಎಳೆಗಳು);

    ಆದರೆ (gctx.layout.ಒಟ್ಟು_ಬೈಟುಗಳು > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    FARF(ಮೇಲಿನ_2, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.ಒಟ್ಟು_ಬೈಟುಗಳು, n_ಎಳೆಗಳು);

    ಆದರೆ (n_ಟೋಕನುಗಳು == 1) {
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, gated_delta_net_f32_tg_thread, &gctx, n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, gated_delta_net_f32_pp_thread, &gctx, n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
