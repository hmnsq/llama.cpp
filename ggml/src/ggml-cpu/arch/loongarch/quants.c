#నిర్వచించు GGML_COMMON_IMPL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "simd-mappings.h"

#అన్తర్భావ "../../quants.h"
#అన్తర్భావ "../../ggml-cpu-impl.h"

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <float.h>
#అన్తర్భావ <stdlib.h> // for qsort
#అన్తర్భావ <stdio.h>  // for GGML_ASSERT

#నిర్వచించు సమూహం_MAX_EPS 1e-15f
#నిర్వచించు సమూహం_MAX_EPS_IQ3_XXS 1e-8f
#నిర్వచించు సమూహం_MAX_EPS_IQ2_S 1e-8f
#నిర్వచించు సమూహం_MAX_EPS_IQ1_M 1e-7f
#నిర్వచించు సమూహం_MAX_EPS_IQ1_S 1e-12f

#నిర్వచించు వాడని GGML_వాడని

#ఒకవేళ defined(__loongarch_sx)

స్థిర __m128i lsx_packs_w(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_w(a, 15);
    tmp1 = __lsx_vsat_w(b, 15);
    తిరిగి __lsx_vpickev_h(tmp1, tmp);
}

స్థిర __m128i lsx_packs_h(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_h(a, 7);
    tmp1 = __lsx_vsat_h(b, 7);
    తిరిగి __lsx_vpickev_b(tmp1, tmp);
}

స్థిర __m128i lsx_packus_h(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_hu(a, 7);
    tmp1 = __lsx_vsat_hu(b, 7);
    తిరిగి __lsx_vpickev_b(tmp1, tmp);
}

స్థిర __m128i lsx_maddubs_h(__m128i a, __m128i b) {
    __m128i tmp1, tmp2;
    tmp1 = __lsx_vmulwev_h_b(a, b);
    tmp2 = __lsx_vmulwod_h_b(a, b);
    తిరిగి __lsx_vsadd_h(tmp1, tmp2);
}

స్థిర __m128i lsx_madd_h(__m128i a, __m128i b) {
    __m128i tmp1, tmp2;
    tmp1 = __lsx_vmulwev_w_h(a, b);
    tmp2 = __lsx_vmulwod_w_h(a, b);
    తిరిగి __lsx_vadd_w(tmp1, tmp2);
}

స్థిర __m128i lsx_సమితి_w(int32_t a, int32_t b, int32_t c, int32_t d) {
    v4i32 __ret = {d, c, b, a};
    తిరిగి (__m128i)__ret;
}

స్థిర __m128i lsx_shuffle_b(__m128i a, __m128i b) {
    __m128i ముసుగు_f, zero, tmp0, tmp2, ముసుగు;
    పూర్ణాంకం f = 0x8f;
    ముసుగు_f = __lsx_vreplgr2vr_b(f);
    zero = __lsx_vldi(0);
    tmp0 = __lsx_vand_v(b, ముసుగు_f); // get mask with low 4 bit and sign bits
    tmp0 = __lsx_vori_b(tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
    ముసుగు = __lsx_vsle_b(zero, tmp0); // if mask >= 0, set mask
    tmp2 = __lsx_vand_v(tmp0, ముసుగు); // maskout the in2 < ones
    తిరిగి __lsx_vshuf_b(a, zero, tmp2);
}

స్థిర __m128i lsx_hadd_h(__m128i a, __m128i b) {
    __m128i tmp1 = __lsx_vpickev_h(b, a);
    __m128i tmp2 = __lsx_vpickod_h(b, a);
    తిరిగి __lsx_vadd_h(tmp1, tmp2);
}

స్థిర __m128i lsx_hadd_w(__m128i a, __m128i b) {
    __m128i tmp1 = __lsx_vpickev_w(b, a);
    __m128i tmp2 = __lsx_vpickod_w(b, a);
    తిరిగి __lsx_vadd_w(tmp1, tmp2);
}

స్థిర __m128 lsx_hadd_s(__m128 a, __m128 b) {
    __m128 tmp1 = (__m128)__lsx_vpickev_w((__m128i)b, (__m128i)a);
    __m128 tmp2 = (__m128)__lsx_vpickod_w((__m128i)b, (__m128i)a);

    తిరిగి __lsx_vfadd_s(tmp1, tmp2);
}

స్థిర అంతఃవరుస దశాంశం hsum_float_4x4(స్థిరరాశి __m128 a, స్థిరరాశి __m128 b, స్థిరరాశి __m128 c, స్థిరరాశి __m128 d) {
    __m128 res_0 =lsx_hadd_s(a, b);
    __m128 res_1 =lsx_hadd_s(c, d);
    __m128 res =lsx_hadd_s(res_0, res_1);
    res =lsx_hadd_s(res, res);
    res =lsx_hadd_s(res, res);

    తిరిగి ((v4f32)res)[0];
}

// multiply int8_t, add results pairwise twice
స్థిర అంతఃవరుస __m128i mul_కూడిక_i8_జతలు(స్థిరరాశి __m128i x, స్థిరరాశి __m128i y) {
    // Get absolute values of x vectors
    స్థిరరాశి __m128i ax = __lsx_vsigncov_b(x, x);
    // Sign the values of the y vectors
    స్థిరరాశి __m128i sy = __lsx_vsigncov_b(x, y);
    // Perform multiplication and create 16-bit values
    స్థిరరాశి __m128i dot = lsx_maddubs_h(ax, sy);
    స్థిరరాశి __m128i ones = __lsx_vreplgr2vr_h(1);
    తిరిగి lsx_madd_h(ones, dot);
}
#అయితే_ముగింపు

#ఒకవేళ defined(__loongarch_asx)

#నిర్వచితమైతే __clang__
#నిర్వచించు VREGS_ఉపసర్గ "$vr"
#నిర్వచించు XREGS_ఉపసర్గ "$xr"
#లేకపోతే // GCC
#నిర్వచించు VREGS_ఉపసర్గ "$f"
#నిర్వచించు XREGS_ఉపసర్గ "$f"
#అయితే_ముగింపు
#నిర్వచించు __ALL_REGS "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
// Convert __m128i to __m256i
స్థిర అంతఃవరుస __m256i ____m256i(__m128i లోపల) {
    __m256i బహిర్గతం = __lasx_xvldi(0);
    __asm__ అస్థిర (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_ఉపసర్గ"\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " VREGS_ఉపసర్గ "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [బహిర్గతం] "+f" (బహిర్గతం) : [లోపల] "f" (లోపల)
    );
    తిరిగి బహిర్గతం;
}
// Convert two __m128i to __m256i
స్థిర అంతఃవరుస __m256i lasx_సమితి_q(__m128i inhi, __m128i inlo) {
    __m256i బహిర్గతం;
    __asm__ అస్థిర (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[hi], " VREGS_ఉపసర్గ "\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[lo], " VREGS_ఉపసర్గ "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".ifnc %[out], %[hi]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_ఉపసర్గ "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[hi], " VREGS_ఉపసర్గ "\\j  \n\t"
        "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [బహిర్గతం] "=f" (బహిర్గతం), [hi] "+f" (inhi)
        : [lo] "f" (inlo)
    );
    తిరిగి బహిర్గతం;
}
// Convert __m256i low part to __m128i
స్థిర అంతఃవరుస __m128i lasx_extracti128_lo(__m256i లోపల) {
    __m128i బహిర్గతం;
    __asm__ అస్థిర (
        ".ifnc %[out], %[in]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_ఉపసర్గ "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_ఉపసర్గ "\\j  \n\t"
        "    vori.b $vr\\i, $vr\\j, 0        \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [బహిర్గతం] "=f" (బహిర్గతం) : [లోపల] "f" (లోపల)
    );
    తిరిగి బహిర్గతం;
}
// Convert __m256i high part to __m128i
స్థిర అంతఃవరుస __m128i lasx_extracti128_hi(__m256i లోపల) {
    __m128i బహిర్గతం;
    __asm__ అస్థిర (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_ఉపసర్గ "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_ఉపసర్గ "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [బహిర్గతం] "=f" (బహిర్గతం) : [లోపల] "f" (లోపల)
    );
    తిరిగి బహిర్గతం;
}

స్థిర __m256i lasx_సమితి_w(పూర్ణాంకం e7, పూర్ణాంకం e6, పూర్ణాంకం e5, పూర్ణాంకం e4, పూర్ణాంకం e3, పూర్ణాంకం e2, పూర్ణాంకం e1, పూర్ణాంకం e0) {
    v8i32 __ret = {e0, e1, e2, e3, e4, e5, e6, e7};
    తిరిగి (__m256i)__ret;
}

స్థిర __m256i lasx_సమితి_d(int64_t a, int64_t b, int64_t c, int64_t d) {
    v4i64 __ret = {d, c, b, a};
    తిరిగి (__m256i)__ret;
}

స్థిర __m256i lasx_insertf128( __m128i x, __m128i y) {
    తిరిగి lasx_సమితి_q(x, y);
}

స్థిర __m256i lasx_shuffle_b(__m256i a, __m256i b) {
    __m256i ముసుగు_f, zero, tmp0, tmp2, ముసుగు;
    పూర్ణాంకం f = 0x8f;
    ముసుగు_f = __lasx_xvreplgr2vr_b(f);
    zero = __lasx_xvldi(0);
    tmp0 = __lasx_xvand_v(b, ముసుగు_f); // get mask with low 4 bit and sign bits
    tmp0 = __lasx_xvori_b(tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
    ముసుగు = __lasx_xvsle_b(zero, tmp0); // if mask >= 0, set mask
    tmp2 = __lasx_xvand_v(tmp0, ముసుగు); // maskout the in2 < ones
    తిరిగి __lasx_xvshuf_b(a, zero, tmp2);
}

స్థిర __m256i lasx_extu8_16(__m128i a) {
    తిరిగి __lasx_vext2xv_hu_bu(____m256i(a));
}

స్థిర __m256i lasx_ext8_16(__m128i a) {
    తిరిగి __lasx_vext2xv_h_b(____m256i(a));
}

స్థిర __m256i lasx_ext16_32(__m128i a) {
    తిరిగి __lasx_vext2xv_w_h(____m256i(a));
}

స్థిర __m128i lasx_extracti128( __m256i a, పూర్ణాంకం pos) {
    __m128i ret;
    ఒకవేళ( pos == 0)
    {
       ret = lasx_extracti128_lo(a);
    } లేకపోతే {
       ret = lasx_extracti128_hi(a);
    }
    తిరిగి ret;
}

స్థిర __m128 lasx_extractf128( __m256 a, పూర్ణాంకం pos) {
    __m128 ret;
    ఒకవేళ( pos == 0)
    {
       ret = (__m128)lasx_extracti128_lo((__m256i)a);
    } లేకపోతే {
       ret = (__m128)lasx_extracti128_hi((__m256i)a);
    }
    తిరిగి ret;
}

స్థిర __m256i lasx_maddubs_h(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_h_b(a, b);
    tmp2 = __lasx_xvmulwod_h_b(a, b);
    తిరిగి __lasx_xvsadd_h(tmp1, tmp2);
}

స్థిర __m256i lasx_madd_h(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_w_h(a, b);
    tmp2 = __lasx_xvmulwod_w_h(a, b);
    తిరిగి __lasx_xvadd_w(tmp1, tmp2);
}

స్థిర __m256i lasx_packs_w(__m256i a, __m256i b) {
    __m256i tmp, tmp1;
    tmp = __lasx_xvsat_w(a, 15);
    tmp1 = __lasx_xvsat_w(b, 15);
    తిరిగి __lasx_xvpickev_h(tmp1, tmp);
}

స్థిర __m256i lasx_packs_h(__m256i a, __m256i b) {
    __m256i tmp, tmp1;
    tmp = __lasx_xvsat_h(a, 7);
    tmp1 = __lasx_xvsat_h(b, 7);
    తిరిగి __lasx_xvpickev_b(tmp1, tmp);
}

స్థిర అంతఃవరుస __m256i lasx_madd_h_b(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_h_b(a, b);
    tmp2 = __lasx_xvmulwod_h_b(a, b);
    తిరిగి __lasx_xvadd_h(tmp1, tmp2);
}

స్థిర అంతఃవరుస __m256i lasx_xvrepl128vei_h(__m256i a, స్థిరరాశి గుర్తులేని పూర్ణాంకం b) {
    ఎంపిక (b) {
        స్థితి 0: తిరిగి __lasx_xvrepl128vei_h(a, 0);
        స్థితి 1: తిరిగి __lasx_xvrepl128vei_h(a, 1);
        స్థితి 2: తిరిగి __lasx_xvrepl128vei_h(a, 2);
        స్థితి 3: తిరిగి __lasx_xvrepl128vei_h(a, 3);
        స్థితి 4: తిరిగి __lasx_xvrepl128vei_h(a, 4);
        స్థితి 5: తిరిగి __lasx_xvrepl128vei_h(a, 5);
        స్థితి 6: తిరిగి __lasx_xvrepl128vei_h(a, 6);
        స్థితి 7: తిరిగి __lasx_xvrepl128vei_h(a, 7);
        అప్రమేయ: __builtin_unreachable();
    }
}

స్థిర అంతఃవరుస __m256i lasx_xvandi_b_బిట్టు(__m256i a, స్థిరరాశి గుర్తులేని పూర్ణాంకం b) {
    ఎంపిక (b) {
        స్థితి 0: తిరిగి __lasx_xvandi_b(a, 1 << 0);
        స్థితి 1: తిరిగి __lasx_xvandi_b(a, 1 << 1);
        స్థితి 2: తిరిగి __lasx_xvandi_b(a, 1 << 2);
        స్థితి 3: తిరిగి __lasx_xvandi_b(a, 1 << 3);
        స్థితి 4: తిరిగి __lasx_xvandi_b(a, 1 << 4);
        స్థితి 5: తిరిగి __lasx_xvandi_b(a, 1 << 5);
        స్థితి 6: తిరిగి __lasx_xvandi_b(a, 1 << 6);
        స్థితి 7: తిరిగి __lasx_xvandi_b(a, 1 << 7);
        అప్రమేయ: __builtin_unreachable();
    }
}

// horizontally add 8 floats
స్థిర అంతఃవరుస దశాంశం hsum_float_8(స్థిరరాశి __m256 x) {
    __m128 res = lasx_extractf128(x, 1);
    res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
    res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
    res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
    తిరిగి ((v4f32)res)[0];
}

// horizontally add 8 int32_t
స్థిర అంతఃవరుస పూర్ణాంకం hsum_i32_8(స్థిరరాశి __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    పూర్ణాంకం sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    తిరిగి  sum64_1 + sum64_2;
}

// horizontally add 4 int32_t
స్థిర అంతఃవరుస పూర్ణాంకం hsum_i32_4(స్థిరరాశి __m128i a) {
    __m128i ev = __lsx_vpickev_w(a, a);
    __m128i od = __lsx_vpickod_w(a, a);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    పూర్ణాంకం sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    తిరిగి  sum64_1 + sum64_2;
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
స్థిర అంతఃవరుస __m256i బైట్లు_from_బిట్లు_32(స్థిరరాశి uint8_t * x) {

    uint32_t x32;
    జ్ఞాపకం_నకలు(&x32, x, పరిమాణం(uint32_t));
    స్థిరరాశి __m256i shuf_ముసుగు = lasx_సమితి_d(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);

    __m256i బైట్లు = lasx_shuffle_b(__lasx_xvreplgr2vr_w(x32), shuf_ముసుగు);
    స్థిరరాశి __m256i బిట్టు_ముసుగు = __lasx_xvreplgr2vr_d(0x7fbfdfeff7fbfdfe);
    బైట్లు = __lasx_xvor_v(బైట్లు, బిట్టు_ముసుగు);
    తిరిగి __lasx_xvseq_b(బైట్లు, __lasx_xvreplgr2vr_d(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
స్థిర అంతఃవరుస __m256i బైట్లు_from_nibbles_32(స్థిరరాశి uint8_t * rsi) {
    స్థిరరాశి __m128i lo = __lsx_vld((స్థిరరాశి __m128i *)rsi, 0);
    __m128i hi = __lsx_vsrli_h(lo, 4);
    తిరిగి __lasx_xvandi_b(lasx_insertf128(hi, lo), 0xf);
}

// add int16_t pairwise and return as float vector
స్థిర అంతఃవరుస __m256 కూడిక_i16_జతలు_float(స్థిరరాశి __m256i x) {
    __m256i v = __lasx_xvpackod_h(x, x);
    __m256i summed_జతలు = __lasx_xvaddwev_w_h(x, v);
    తిరిగి __lasx_xvffint_s_w(summed_జతలు);
}

స్థిర అంతఃవరుస __m256 mul_కూడిక_us8_జతలు_float(స్థిరరాశి __m256i ax, స్థిరరాశి __m256i sy) {
    // Perform multiplication and create 16-bit values
    స్థిరరాశి __m256i dot = lasx_maddubs_h(ax, sy);
    తిరిగి కూడిక_i16_జతలు_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
స్థిర అంతఃవరుస __m256 mul_కూడిక_i8_జతలు_float(స్థిరరాశి __m256i x, స్థిరరాశి __m256i y) {
    స్థిరరాశి __m256i dot = lasx_madd_h_b(x, y);
    తిరిగి కూడిక_i16_జతలు_float(dot);
}

స్థిర అంతఃవరుస __m128i packNibbles( __m256i బైట్లు ) {
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    స్థిరరాశి __m256i దిగువ_Byte = __lasx_xvreplgr2vr_h(0xFF);
     __m256i ఎగువ = __lasx_xvandn_v(దిగువ_Byte, బైట్లు);
    __m256i దిగువ = __lasx_xvand_v(దిగువ_Byte, బైట్లు);
    ఎగువ = __lasx_xvsrli_h(ఎగువ, 4);
    బైట్లు = __lasx_xvor_v(దిగువ, ఎగువ);
    // Compress uint16_t lanes into bytes
    __m128i *r0 = (__m128i *)&బైట్లు;
    __m256i tmp_h128 = __lasx_xvpermi_q(బైట్లు, బైట్లు, 0x11);
    __m128i *r1 = (__m128i *)&tmp_h128;

    __m128i zero = __lsx_vldi(0);
    __m128i tmp, tmp2, tmp3;

    tmp = __lsx_vmax_h(zero, *r0);
    tmp2 = __lsx_vsat_hu(tmp, 7);

    tmp = __lsx_vmax_h(zero, *r1);
    tmp3 = __lsx_vsat_hu(tmp, 7);
    తిరిగి  __lsx_vpickev_b(tmp3, tmp2);
}
#అయితే_ముగింపు  //__loongarch_asx

శూన్య quantize_అడ్డువరుస_q8_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0 * GGML_RESTRICT y = vy;

#ఒకవేళ defined(__loongarch_asx)
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        __m256 v0 = (__m256)__lasx_xvld( x , 0);
        __m256 v1 = (__m256)__lasx_xvld( x , 32);
        __m256 v2 = (__m256)__lasx_xvld( x , 64);
        __m256 v3 = (__m256)__lasx_xvld( x , 96);
        x += 32;

        // Compute max(abs(e)) for the block
        స్థిరరాశి __m256 sign_బిట్టు_2 = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs , 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w( max4, 1 ), 0 ));
        స్థిరరాశి దశాంశం max_scalar = ((v4f32)max4)[0];

        // Quantize these floats
        స్థిరరాశి దశాంశం d = max_scalar / 127.f;
        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);
        స్థిరరాశి దశాంశం id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        స్థిరరాశి __m256 mul = (__m256)__lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128( i0, 0 );
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0);
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);

    }
#లేకపోతే
    GGML_వాడని(nb);
    // scalar
    quantize_అడ్డువరుస_q8_0_ref(x, y, k);
#అయితే_ముగింపు
}

శూన్య quantize_అడ్డువరుస_q8_1(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(k % QK8_1 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_1;

    ఖండం_q8_1 * GGML_RESTRICT y = vy;

#ఒకవేళ defined(__loongarch_asx)
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        __m256 v0 = (__m256)__lasx_xvld( x , 0 );
        __m256 v1 = (__m256)__lasx_xvld( x , 32 );
        __m256 v2 = (__m256)__lasx_xvld( x , 64 );
        __m256 v3 = (__m256)__lasx_xvld( x , 96 );
        x += 32;

        // Compute max(abs(e)) for the block
        స్థిరరాశి __m256 sign_బిట్టు_2 = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_బిట్టు_2, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs, 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vextrins_w((__m128i)tmp, (__m128i)max4, 0x1 ));
        స్థిరరాశి దశాంశం max_scalar = ((v4f32)max4)[0];

        // Quantize these floats
        స్థిరరాశి దశాంశం d = max_scalar / 127.f;
        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);
        స్థిరరాశి దశాంశం id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        స్థిరరాశి __m256 mul = __lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128(i0, 0);
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0 );
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Compute the sum of the quants and set y[i].s
        స్థిరరాశి __m128i s0 = __lsx_vadd_w(__lsx_vadd_w(ni0, ni1), __lsx_vadd_w(ni2, ni3));
        స్థిరరాశి __m128i s1 = __lsx_vadd_w(__lsx_vadd_w(ni4, ni5), __lsx_vadd_w(ni6, ni7));
        y[i].s = GGML_CPU_FP32_గమ్యం_FP16(d * hsum_i32_4(__lsx_vadd_w(s0, s1)));

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);
    }
#లేకపోతే
    GGML_వాడని(nb);
    // scalar
    quantize_అడ్డువరుస_q8_1_ref(x, y, k);
#అయితే_ముగింపు
}


//===================================== Dot products =================================

//
// Helper functions
//

#ఒకవేళ defined(__loongarch_asx)
// shuffles to pick the required scales in dot products
స్థిర అంతఃవరుస __m256i get_scale_shuffle_q3k(పూర్ణాంకం i) {
    స్థిర స్థిరరాశి uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    తిరిగి __lasx_xvld((స్థిరరాశి __m256i*)k_shuffle + i, 0);
}
స్థిర అంతఃవరుస __m256i get_scale_shuffle_k4(పూర్ణాంకం i) {
    స్థిర స్థిరరాశి uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    తిరిగి __lasx_xvld((స్థిరరాశి __m256i*)k_shuffle + i, 0);
}
స్థిర అంతఃవరుస __m128i get_scale_shuffle(పూర్ణాంకం i) {
    స్థిర స్థిరరాశి uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    తిరిగి __lsx_vld((స్థిరరాశి __m128i*)k_shuffle + i, 0);
}
#అయితే_ముగింపు

శూన్య ggml_vec_dot_q4_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q4_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    కొరకు (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        స్థిరరాశి __m256 d = __lasx_xvreplfr2vr_s( GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d) );

        __m256i qx = బైట్లు_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        స్థిరరాశి __m256i off = __lasx_xvreplgr2vr_b( 8 );
        qx = __lasx_xvsub_b( qx, off );

        __m256i qy = __lasx_xvld((స్థిరరాశి __m256i *)y[ib].qs, 0);

        స్థిరరాశి __m256 q = mul_కూడిక_i8_జతలు_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);

#లేదావేళ defined(__loongarch_sx)
    // set constants
    స్థిరరాశి __m128i దిగువ_ముసుగు_2 = __lsx_vreplgr2vr_b(0xF);
    స్థిరరాశి __m128i off = __lsx_vreplgr2vr_b(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = (__m128)__lsx_vldi(0);
    __m128 acc_1 = (__m128)__lsx_vldi(0);
    __m128 acc_2 = (__m128)__lsx_vldi(0);
    __m128 acc_3 = (__m128)__lsx_vldi(0);

    కొరకు (; ib + 1 < nb; ib += 2) {

        // Compute combined scale for the block 0 and 1
        స్థిరరాశి దశాంశం ft0 = GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d);
        స్థిరరాశి __m128 d_0_1 = (__m128)(v4f32){ft0, ft0, ft0, ft0};

        స్థిరరాశి __m128i tmp_0_1 = __lsx_vld((స్థిరరాశి __m128i *)x[ib].qs, 0);

        __m128i bx_0 = __lsx_vand_v(దిగువ_ముసుగు_2, tmp_0_1);
        __m128i by_0 = __lsx_vld((స్థిరరాశి __m128i *)y[ib].qs, 0);
        bx_0 = __lsx_vsub_b(bx_0, off);
        స్థిరరాశి __m128i i32_0 = mul_కూడిక_i8_జతలు(bx_0, by_0);

        __m128i bx_1 = __lsx_vand_v(దిగువ_ముసుగు_2, __lsx_vsrli_d(tmp_0_1, 4));
        __m128i by_1 = __lsx_vld((స్థిరరాశి __m128i *)(y[ib].qs + 16), 0);
        bx_1 = __lsx_vsub_b(bx_1, off);
        స్థిరరాశి __m128i i32_1 = mul_కూడిక_i8_జతలు(bx_1, by_1);

        // Compute combined scale for the block 2 and 3
        స్థిరరాశి దశాంశం ft1 = GGML_CPU_FP16_గమ్యం_FP32(x[ib + 1].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib + 1].d);
        స్థిరరాశి __m128 d_2_3 = (__m128)(v4f32){ft1, ft1, ft1, ft1};

        స్థిరరాశి __m128i tmp_2_3 = __lsx_vld((స్థిరరాశి __m128i *)x[ib + 1].qs, 0);

        __m128i bx_2 = __lsx_vand_v(దిగువ_ముసుగు_2, tmp_2_3);
        __m128i by_2 = __lsx_vld((స్థిరరాశి __m128i *)y[ib + 1].qs, 0);
        bx_2 = __lsx_vsub_b(bx_2, off);
        స్థిరరాశి __m128i i32_2 = mul_కూడిక_i8_జతలు(bx_2, by_2);

        __m128i bx_3 = __lsx_vand_v(దిగువ_ముసుగు_2, __lsx_vsrli_d(tmp_2_3, 4));
        __m128i by_3 = __lsx_vld((స్థిరరాశి __m128i *)(y[ib + 1].qs + 16), 0);
        bx_3 = __lsx_vsub_b(bx_3, off);
        స్థిరరాశి __m128i i32_3 = mul_కూడిక_i8_జతలు(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = __lsx_vffint_s_w(i32_0);
        __m128 p1 = __lsx_vffint_s_w(i32_1);
        __m128 p2 = __lsx_vffint_s_w(i32_2);
        __m128 p3 = __lsx_vffint_s_w(i32_3);

        // Apply the scale
        __m128 p0_d = __lsx_vfmul_s( d_0_1, p0 );
        __m128 p1_d = __lsx_vfmul_s( d_0_1, p1 );
        __m128 p2_d = __lsx_vfmul_s( d_2_3, p2 );
        __m128 p3_d = __lsx_vfmul_s( d_2_3, p3 );

        // Acummulate
        acc_0 = __lsx_vfadd_s(p0_d, acc_0);
        acc_1 = __lsx_vfadd_s(p1_d, acc_1);
        acc_2 = __lsx_vfadd_s(p2_d, acc_2);
        acc_3 = __lsx_vfadd_s(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);

#అయితే_ముగింపు
    కొరకు (; ib < nb; ++ib) {
        పూర్ణాంకం sumi0 = 0;
        పూర్ణాంకం sumi1 = 0;

        కొరకు (పూర్ణాంకం j = 0; j < qk/2; ++j) {
            స్థిరరాశి పూర్ణాంకం v0 = (x[ib].qs[j] & 0x0F) - 8;
            స్థిరరాశి పూర్ణాంకం v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        పూర్ణాంకం sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_గమ్యం_FP32(x[ib].d)*GGML_CPU_FP16_గమ్యం_FP32(y[ib].d);
    }

    *s = sumf;
}

శూన్య ggml_vec_dot_q4_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_1;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q4_1 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    దశాంశం summs = 0;

    // Main loop
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి దశాంశం d0 = GGML_CPU_FP16_గమ్యం_FP32(x[ib].d);
        స్థిరరాశి దశాంశం d1 = GGML_CPU_FP16_గమ్యం_FP32(y[ib].d);

        summs += GGML_CPU_FP16_గమ్యం_FP32(x[ib].m) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].s);

        స్థిరరాశి __m256 d0v = __lasx_xvreplfr2vr_s( d0 );
        స్థిరరాశి __m256 d1v = __lasx_xvreplfr2vr_s( d1 );

        // Compute combined scales
        స్థిరరాశి __m256 d0d1 = __lasx_xvfmul_s( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        స్థిరరాశి __m256i qx = బైట్లు_from_nibbles_32(x[ib].qs);
        స్థిరరాశి __m256i qy = __lasx_xvld( (స్థిరరాశి __m256i *)y[ib].qs, 0);

        స్థిరరాశి __m256 xy = mul_కూడిక_us8_జతలు_float(qx, qy);

        // Accumulate d0*d1*x*y
        acc = __lasx_xvfmadd_s( d0d1, xy, acc );
    }

    sumf = hsum_float_8(acc) + summs;

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(qk == QK5_0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

#ఒకవేళ defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    కొరకు (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        స్థిరరాశి __m256 d = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d)); //FIXME

        __m256i qx = బైట్లు_from_nibbles_32(x[ib].qs);
        __m256i bxhi = బైట్లు_from_బిట్లు_32(x[ib].qh);
        bxhi = __lasx_xvandn_v(bxhi, __lasx_xvreplgr2vr_b((అక్షరం)0xF0));
        qx = __lasx_xvor_v(qx, bxhi);

        __m256i qy = __lasx_xvld((స్థిరరాశి __m256i *)y[ib].qs, 0);

        స్థిరరాశి __m256 q = mul_కూడిక_i8_జతలు_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s(d, q, acc);
    }

    sumf = hsum_float_8(acc);

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(ib);
    వాడని(sumf);
    వాడని(x);
    వాడని(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_1;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(qk == QK5_1);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y = vy;

#ఒకవేళ defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    దశాంశం summs = 0.0f;

    // Main loop
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి __m256 dx = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));

        summs += GGML_CPU_FP16_గమ్యం_FP32(x[ib].m) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].s);

        __m256i qx = బైట్లు_from_nibbles_32(x[ib].qs);
        __m256i bxhi = బైట్లు_from_బిట్లు_32(x[ib].qh);
        bxhi = __lasx_xvand_v(bxhi, __lasx_xvreplgr2vr_b(0x10));
        qx = __lasx_xvor_v(qx, bxhi);

        స్థిరరాశి __m256 dy = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        స్థిరరాశి __m256i qy = __lasx_xvld((స్థిరరాశి __m256i *)y[ib].qs, 0);

        స్థిరరాశి __m256 q = mul_కూడిక_us8_జతలు_float(qx, qy);

        acc = __lasx_xvfmadd_s(q, __lasx_xvfmul_s(dx, dy), acc);
    }

    sumf = hsum_float_8(acc) + summs;

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(ib);
    వాడని(sumf);
    వాడని(x);
    వాడని(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q8_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    కొరకు (; ib < nb; ++ib) {
        // Compute combined scale for the block
        స్థిరరాశి __m256 d = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        __m256i qx = __lasx_xvld((స్థిరరాశి __m256i *)x[ib].qs, 0);
        __m256i qy = __lasx_xvld((స్థిరరాశి __m256i *)y[ib].qs, 0);

        స్థిరరాశి __m256 q = mul_కూడిక_i8_జతలు_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);

    *s = sumf;

#లేదావేళ defined(__loongarch_sx)

    __m128 acc = (__m128)__lsx_vldi(0);

    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d);
        స్థిరరాశి __m128i qx_0 = __lsx_vld((స్థిరరాశి __m128i *)x[ib].qs, 0);
        స్థిరరాశి __m128i qx_1 = __lsx_vld((స్థిరరాశి __m128i *)x[ib].qs + 1, 0);
        స్థిరరాశి __m128i qy_0 = __lsx_vld((స్థిరరాశి __m128i *)y[ib].qs, 0);
        స్థిరరాశి __m128i qy_1 = __lsx_vld((స్థిరరాశి __m128i *)y[ib].qs + 1, 0);

        స్థిరరాశి __m128i p16_0 = lsx_maddubs_h(qx_0, qy_0);
        స్థిరరాశి __m128i p16_1 = lsx_maddubs_h(qx_1, qy_1);

        // Sum int16 pairs → int32
        స్థిరరాశి __m128i s_0 = __lsx_vaddwev_w_h(p16_0, p16_1);
        స్థిరరాశి __m128i s_1 = __lsx_vaddwod_w_h(p16_0, p16_1);

        స్థిరరాశి __m128 q = __lsx_vffint_s_w(__lsx_vadd_w(s_0, s_1));
        acc = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(d), q, acc);
    }

    __m128 res = lsx_hadd_s(acc, acc);
    res = lsx_hadd_s(res, res);
    sumf = ((v4f32)res)[0];

    *s = sumf;

#లేకపోతే
    వాడని(nb);
    వాడని(ib);
    వాడని(sumf);
    వాడని(x);
    వాడని(y);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q2_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q2_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = -y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin);

        స్థిరరాశి uint8_t * GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        స్థిరరాశి __m128i mins_and_scales128 = __lsx_vld((స్థిరరాశి __m128i*)x[i].scales, 0);
        స్థిరరాశి __m128i scales128 = __lsx_vandi_b(mins_and_scales128, 0xf);
        స్థిరరాశి __m256i mins = lasx_ext8_16(__lsx_vsrli_b(mins_and_scales128, 4));
        స్థిరరాశి __m256i prod = lasx_madd_h(mins, __lasx_xvld((స్థిరరాశి __m256i*)y[i].bsums, 0));

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(dmin), __lasx_xvffint_s_w(prod), acc);

        స్థిరరాశి v16i8 shuffle_ముసుగు = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        స్థిరరాశి __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_ముసుగు));

        __m256i sumi = __lasx_xvldi(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {

            స్థిరరాశి __m256i q2bits = __lasx_xvld((స్థిరరాశి __m256i*)q2, 0); q2 += 32;

            స్థిరరాశి __m256i q8_0 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_3 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;

            స్థిరరాశి __m256i q2_0 = __lasx_xvandi_b(q2bits, 3);
            స్థిరరాశి __m256i q2_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 2), 3);
            స్థిరరాశి __m256i q2_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 4), 3);
            స్థిరరాశి __m256i q2_3 = __lasx_xvsrli_b(q2bits, 6);

            __m256i p0 = lasx_madd_h_b(q2_0, q8_0);
            __m256i p1 = lasx_madd_h_b(q2_1, q8_1);
            __m256i p2 = lasx_madd_h_b(q2_2, q8_2);
            __m256i p3 = lasx_madd_h_b(q2_3, q8_3);

            p0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p0);
            p1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p1);
            p2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p2);
            p3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p3);

            p0 = __lasx_xvadd_w(p0, p1);
            p2 = __lasx_xvadd_w(p2, p3);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p0, p2));
        }

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);

    }

    *s = hsum_float_8(acc);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q3_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి uint32_t kmask1 = 0x03030303;
    స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;

    స్థిరరాశి ఖండం_q3_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined __loongarch_asx

    స్థిరరాశి __m128i m32 = __lsx_vreplgr2vr_b(32);

    __m256 acc = (__m256)__lasx_xvldi(0);

    uint32_t aux[3];

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి uint8_t * GGML_RESTRICT q3 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;
        // Set up scales
        జ్ఞాపకం_నకలు(aux, x[i].scales, 12);
        __m128i scales128 = lsx_సమితి_w(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = __lsx_vsub_b(scales128, m32);

        స్థిరరాశి v16i8 shuffle_ముసుగు = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        స్థిరరాశి __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_ముసుగు));

        // high bit
        స్థిరరాశి __m256i hbits = __lasx_xvld((స్థిరరాశి __m256i*)x[i].hmask, 0);

        // integer accumulator
        __m256i sumi = __lasx_xvldi(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            స్థిరరాశి __m256i q3bits = __lasx_xvld((స్థిరరాశి __m256i*)q3, 0); q3 += 32;

            // prepare low and high bits
            స్థిరరాశి __m256i q3l_0 = __lasx_xvandi_b(q3bits, 3);
            స్థిరరాశి __m256i q3l_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 2), 3);
            స్థిరరాశి __m256i q3l_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 4), 3);
            స్థిరరాశి __m256i q3l_3 = __lasx_xvsrli_b(q3bits, 6);
            స్థిరరాశి __m256i q3h_0 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_బిట్టు(hbits, 4 * j + 0), 0), 2);
            స్థిరరాశి __m256i q3h_1 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_బిట్టు(hbits, 4 * j + 1), 0), 2);
            స్థిరరాశి __m256i q3h_2 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_బిట్టు(hbits, 4 * j + 2), 0), 2);
            స్థిరరాశి __m256i q3h_3 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_బిట్టు(hbits, 4 * j + 3), 0), 2);
            స్థిరరాశి __m256i q3_0 = __lasx_xvor_v(q3h_0, q3l_0);
            స్థిరరాశి __m256i q3_1 = __lasx_xvor_v(q3h_1, q3l_1);
            స్థిరరాశి __m256i q3_2 = __lasx_xvor_v(q3h_2, q3l_2);
            స్థిరరాశి __m256i q3_3 = __lasx_xvor_v(q3h_3, q3l_3);

            // load Q8 quants
            స్థిరరాశి __m256i q8_0 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_3 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;

            __m256i p16_0 = lasx_madd_h_b(q8_0, q3_0);
            __m256i p16_1 = lasx_madd_h_b(q8_1, q3_1);
            __m256i p16_2 = lasx_madd_h_b(q8_2, q3_2);
            __m256i p16_3 = lasx_madd_h_b(q8_3, q3_3);

            // multiply with scales
            p16_0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p16_0);
            p16_1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p16_1);
            p16_2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p16_2);
            p16_3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p16_3);

            // accumulate
            p16_0 = __lasx_xvadd_w(p16_0, p16_1);
            p16_2 = __lasx_xvadd_w(p16_2, p16_3);
            sumi  = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_2));
        }
        // multiply with block scale and accumulate
        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
    }

    *s = hsum_float_8(acc);

#లేకపోతే
    వాడని(kmask1);
    వాడని(kmask2);
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q4_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q4_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ఒకవేళ defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);
    __m128 acc_m = (__m128)__lsx_vldi(0);

   కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = -y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin);

        జ్ఞాపకం_నకలు(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        స్థిరరాశి __m128i mins_and_scales128 = lsx_సమితి_w(utmp[3], utmp[2], utmp[1], utmp[0]);
        స్థిరరాశి __m128i mins128 = __lsx_vexth_h_b(mins_and_scales128);
        స్థిరరాశి __m128i scales128 = __lsx_vsllwil_h_b(mins_and_scales128, 0);

        స్థిరరాశి __m256i q8sums = __lasx_xvld((స్థిరరాశి __m256i*)y[i].bsums, 0);
        స్థిరరాశి __m128i q8s = lsx_hadd_h(lasx_extracti128(q8sums, 0), lasx_extracti128(q8sums, 1));
        స్థిరరాశి __m128i prod = lsx_madd_h(mins128, q8s);
        acc_m = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(dmin), __lsx_vffint_s_w(prod), acc_m);

        స్థిరరాశి __m256i scales = lasx_insertf128(scales128, scales128);

        __m256i sumi = __lasx_xvldi(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {

            స్థిరరాశి __m256i scale_l = lasx_xvrepl128vei_h(scales, 2 * j + 0);
            స్థిరరాశి __m256i scale_h = lasx_xvrepl128vei_h(scales, 2 * j + 1);

            స్థిరరాశి __m256i q4bits = __lasx_xvld((స్థిరరాశి __m256i*)q4, 0); q4 += 32;
            స్థిరరాశి __m256i q4l = __lasx_xvandi_b(q4bits, 0xf);
            స్థిరరాశి __m256i q4h = __lasx_xvsrli_b(q4bits, 4);

            స్థిరరాశి __m256i q8l = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            __m256i p16l = lasx_madd_h_b(q4l, q8l);
            p16l = lasx_madd_h(scale_l, p16l);

            స్థిరరాశి __m256i q8h = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            __m256i p16h = lasx_madd_h_b(q4h, q8h);
            p16h = lasx_madd_h(scale_h, p16h);
            స్థిరరాశి __m256i sumj = __lasx_xvadd_w(p16l, p16h);

            sumi = __lasx_xvadd_w(sumi, sumj);
        }

        __m256 vd = __lasx_xvreplfr2vr_s(d);
        acc = __lasx_xvfmadd_s(vd, __lasx_xvffint_s_w(sumi), acc);

    }

    acc_m = __lsx_vfadd_s(acc_m, (__m128)__lsx_vpermi_w((__m128i)acc_m, (__m128i)acc_m, 0xee));
    __m128i tmp1 = __lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w((__m128i)acc_m, 1), 0);
    acc_m = __lsx_vfadd_s(acc_m, (__m128)tmp1);


    *s = hsum_float_8(acc) + ((v4f32)acc_m)[0];

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    వాడని(kmask1);
    వాడని(kmask2);
    వాడని(kmask3);
    వాడని(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy,  size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ఒకవేళ defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);
    __m128 acc_m = (__m128)__lsx_vldi(0);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి uint8_t * GGML_RESTRICT q5 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = -y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin);

        జ్ఞాపకం_నకలు(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        స్థిరరాశి __m128i mins_and_scales128 = lsx_సమితి_w(utmp[3], utmp[2], utmp[1], utmp[0]);
        స్థిరరాశి __m128i mins128 = __lsx_vexth_h_b(mins_and_scales128);
        స్థిరరాశి __m128i scales128 = __lsx_vsllwil_h_b(mins_and_scales128, 0);

        స్థిరరాశి __m256i q8sums = __lasx_xvld((స్థిరరాశి __m256i*)y[i].bsums, 0);
        స్థిరరాశి __m128i q8s = lsx_hadd_h(lasx_extracti128(q8sums, 0), lasx_extracti128(q8sums, 1));
        స్థిరరాశి __m128i prod = lsx_madd_h(mins128, q8s);
        acc_m = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(dmin), __lsx_vffint_s_w(prod), acc_m);

        స్థిరరాశి __m256i scales = lasx_insertf128(scales128, scales128);

        స్థిరరాశి __m256i hbits = __lasx_xvld((స్థిరరాశి __m256i*)x[i].qh, 0);

        __m256i sumi = __lasx_xvldi(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {

            స్థిరరాశి __m256i scale_0 = lasx_xvrepl128vei_h(scales, 2 * j + 0);
            స్థిరరాశి __m256i scale_1 = lasx_xvrepl128vei_h(scales, 2 * j + 1);

            స్థిరరాశి __m256i q5bits = __lasx_xvld((స్థిరరాశి __m256i*)q5, 0); q5 += 32;

            స్థిరరాశి __m256i q5l_0 = __lasx_xvandi_b(q5bits, 0xf);
            స్థిరరాశి __m256i q5l_1 = __lasx_xvsrli_b(q5bits, 4);
            స్థిరరాశి __m256i q5h_0 = __lasx_xvnori_b(__lasx_xvseqi_b(lasx_xvandi_b_బిట్టు(hbits, 2 * j + 0), 0), 0xef);
            స్థిరరాశి __m256i q5h_1 = __lasx_xvnori_b(__lasx_xvseqi_b(lasx_xvandi_b_బిట్టు(hbits, 2 * j + 1), 0), 0xef);
            స్థిరరాశి __m256i q5_0  = __lasx_xvor_v(q5l_0, q5h_0);
            స్థిరరాశి __m256i q5_1  = __lasx_xvor_v(q5l_1, q5h_1);

            స్థిరరాశి __m256i q8_0 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;

            __m256i p16_0 = lasx_madd_h_b(q5_0, q8_0);
            __m256i p16_1 = lasx_madd_h_b(q5_1, q8_1);

            p16_0 = lasx_madd_h(scale_0, p16_0);
            p16_1 = lasx_madd_h(scale_1, p16_1);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_1));

        }

        __m256 vd = __lasx_xvreplfr2vr_s(d);
        acc = __lasx_xvfmadd_s(vd, __lasx_xvffint_s_w(sumi), acc);

    }

    acc_m = __lsx_vfadd_s(acc_m, (__m128)__lsx_vbsrl_v(acc_m, 8));
    acc_m = __lsx_vfadd_s(acc_m, (__m128)__lsx_vbsrl_v(acc_m, 4));

    *s = hsum_float_8(acc) + ((v4f32)acc_m)[0];

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    వాడని(kmask1);
    వాడని(kmask2);
    వాడని(kmask3);
    వాడని(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q6_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q6_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined __loongarch_asx

    స్థిరరాశి __m256i m32s = __lasx_xvreplgr2vr_b(32);

    __m256 acc = (__m256)__lasx_xvldi(0);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);

        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[i].ql;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        స్థిరరాశి __m128i scales128 = __lsx_vld((స్థిరరాశి __m128i*)x[i].scales, 0);
        స్థిరరాశి v16i8 shuffle_ముసుగు = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        స్థిరరాశి __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_ముసుగు));

        __m256i sumi = __lasx_xvldi(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {

            స్థిరరాశి __m256i q4bits1 = __lasx_xvld((స్థిరరాశి __m256i*)q4, 0); q4 += 32;
            స్థిరరాశి __m256i q4bits2 = __lasx_xvld((స్థిరరాశి __m256i*)q4, 0); q4 += 32;
            స్థిరరాశి __m256i q4bitsH = __lasx_xvld((స్థిరరాశి __m256i*)qh, 0); qh += 32;

            స్థిరరాశి __m256i q4h_0 = __lasx_xvslli_b(__lasx_xvandi_b(q4bitsH, 3), 4);
            స్థిరరాశి __m256i q4h_1 = __lasx_xvslli_b(__lasx_xvandi_b(q4bitsH, 3 << 2), 2);
            స్థిరరాశి __m256i q4h_2 = __lasx_xvandi_b(q4bitsH, 3 << 4);
            స్థిరరాశి __m256i q4h_3 = __lasx_xvsrli_b(__lasx_xvandi_b(q4bitsH, 3 << 6), 2);

            స్థిరరాశి __m256i q4_0 = __lasx_xvor_v(__lasx_xvandi_b(q4bits1, 0xf), q4h_0);
            స్థిరరాశి __m256i q4_1 = __lasx_xvor_v(__lasx_xvandi_b(q4bits2, 0xf), q4h_1);
            స్థిరరాశి __m256i q4_2 = __lasx_xvor_v(__lasx_xvsrli_b(q4bits1, 4), q4h_2);
            స్థిరరాశి __m256i q4_3 = __lasx_xvor_v(__lasx_xvsrli_b(q4bits2, 4), q4h_3);

            స్థిరరాశి __m256i q8_0 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_3 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;

            __m256i p16_0 = lasx_madd_h_b(__lasx_xvsub_b(q4_0, m32s), q8_0);
            __m256i p16_1 = lasx_madd_h_b(__lasx_xvsub_b(q4_1, m32s), q8_1);
            __m256i p16_2 = lasx_madd_h_b(__lasx_xvsub_b(q4_2, m32s), q8_2);
            __m256i p16_3 = lasx_madd_h_b(__lasx_xvsub_b(q4_3, m32s), q8_3);

            p16_0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p16_0);
            p16_1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p16_1);
            p16_2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p16_2);
            p16_3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p16_3);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_1));
            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_2, p16_3));
        }

        acc = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
    }

    *s = hsum_float_8(acc);

#లేదావేళ defined(__loongarch_sx)

    స్థిరరాశి __m128i m32s = __lsx_vreplgr2vr_b(32);

    __m128 acc_0 = (__m128)__lsx_vldi(0);
    __m128 acc_1 = (__m128)__lsx_vldi(0);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);

        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[i].ql;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        స్థిరరాశి __m128i scale_i8 = __lsx_vld(x[i].scales, 0);
        స్థిరరాశి __m128i scales_lo = __lsx_vsllwil_h_b(scale_i8, 0);
        స్థిరరాశి __m128i scales_hi = __lsx_vsllwil_h_b(__lsx_vbsrl_v(scale_i8, 8), 0);

        __m128i sumi_0 = __lsx_vldi(0);
        __m128i sumi_1 = __lsx_vldi(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {

            స్థిరరాశి __m128i q4bitsH_0 = __lsx_vld((స్థిరరాశి __m128i*)qh, 0); qh += 16;
            స్థిరరాశి __m128i q4bitsH_1 = __lsx_vld((స్థిరరాశి __m128i*)qh, 0); qh += 16;

            స్థిరరాశి __m128i q4h_0 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_0, 3), 4);
            స్థిరరాశి __m128i q4h_1 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_1, 3), 4);
            స్థిరరాశి __m128i q4h_2 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_0, 3 << 2), 2);
            స్థిరరాశి __m128i q4h_3 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_1, 3 << 2), 2);
            స్థిరరాశి __m128i q4h_4 = __lsx_vandi_b(q4bitsH_0, 3 << 4);
            స్థిరరాశి __m128i q4h_5 = __lsx_vandi_b(q4bitsH_1, 3 << 4);
            స్థిరరాశి __m128i q4h_6 = __lsx_vsrli_b(__lsx_vandi_b(q4bitsH_0, 3 << 6), 2);
            స్థిరరాశి __m128i q4h_7 = __lsx_vsrli_b(__lsx_vandi_b(q4bitsH_1, 3 << 6), 2);

            స్థిరరాశి __m128i q4bits1_0 = __lsx_vld((స్థిరరాశి __m128i*)q4, 0); q4 += 16;
            స్థిరరాశి __m128i q4bits1_1 = __lsx_vld((స్థిరరాశి __m128i*)q4, 0); q4 += 16;
            స్థిరరాశి __m128i q4bits2_0 = __lsx_vld((స్థిరరాశి __m128i*)q4, 0); q4 += 16;
            స్థిరరాశి __m128i q4bits2_1 = __lsx_vld((స్థిరరాశి __m128i*)q4, 0); q4 += 16;

            స్థిరరాశి __m128i q4_0 = __lsx_vor_v(__lsx_vandi_b(q4bits1_0, 0xf), q4h_0);
            స్థిరరాశి __m128i q4_1 = __lsx_vor_v(__lsx_vandi_b(q4bits1_1, 0xf), q4h_1);
            స్థిరరాశి __m128i q4_2 = __lsx_vor_v(__lsx_vandi_b(q4bits2_0, 0xf), q4h_2);
            స్థిరరాశి __m128i q4_3 = __lsx_vor_v(__lsx_vandi_b(q4bits2_1, 0xf), q4h_3);
            స్థిరరాశి __m128i q4_4 = __lsx_vor_v(__lsx_vsrli_b(q4bits1_0, 4), q4h_4);
            స్థిరరాశి __m128i q4_5 = __lsx_vor_v(__lsx_vsrli_b(q4bits1_1, 4), q4h_5);
            స్థిరరాశి __m128i q4_6 = __lsx_vor_v(__lsx_vsrli_b(q4bits2_0, 4), q4h_6);
            స్థిరరాశి __m128i q4_7 = __lsx_vor_v(__lsx_vsrli_b(q4bits2_1, 4), q4h_7);

            స్థిరరాశి __m128i q8_0 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_1 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_2 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_3 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_4 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_5 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_6 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8_7 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;

            __m128i p16_0 = lsx_maddubs_h(__lsx_vsub_b(q4_0, m32s), q8_0);
            __m128i p16_1 = lsx_maddubs_h(__lsx_vsub_b(q4_1, m32s), q8_1);
            __m128i p16_2 = lsx_maddubs_h(__lsx_vsub_b(q4_2, m32s), q8_2);
            __m128i p16_3 = lsx_maddubs_h(__lsx_vsub_b(q4_3, m32s), q8_3);
            __m128i p16_4 = lsx_maddubs_h(__lsx_vsub_b(q4_4, m32s), q8_4);
            __m128i p16_5 = lsx_maddubs_h(__lsx_vsub_b(q4_5, m32s), q8_5);
            __m128i p16_6 = lsx_maddubs_h(__lsx_vsub_b(q4_6, m32s), q8_6);
            __m128i p16_7 = lsx_maddubs_h(__lsx_vsub_b(q4_7, m32s), q8_7);

            స్థిరరాశి __m128i sc_vec = j == 0 ? scales_lo : scales_hi;

            p16_0 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 0), p16_0);
            p16_1 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 1), p16_1);
            p16_2 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 2), p16_2);
            p16_3 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 3), p16_3);
            p16_4 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 4), p16_4);
            p16_5 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 5), p16_5);
            p16_6 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 6), p16_6);
            p16_7 = lsx_madd_h(__lsx_vreplvei_h(sc_vec, 7), p16_7);

            sumi_0 = __lsx_vadd_w(sumi_0, __lsx_vadd_w(p16_0, p16_2));
            sumi_1 = __lsx_vadd_w(sumi_1, __lsx_vadd_w(p16_1, p16_3));
            sumi_0 = __lsx_vadd_w(sumi_0, __lsx_vadd_w(p16_4, p16_6));
            sumi_1 = __lsx_vadd_w(sumi_1, __lsx_vadd_w(p16_5, p16_7));
        }

        __m128 p_0 = __lsx_vfmul_s(__lsx_vreplfr2vr_s(d), __lsx_vffint_s_w(sumi_0));
        __m128 p_1 = __lsx_vfmul_s(__lsx_vreplfr2vr_s(d), __lsx_vffint_s_w(sumi_1));
        acc_0 = __lsx_vfadd_s(p_0, acc_0);
        acc_1 = __lsx_vfadd_s(p_1, acc_1);
    }

    *s = hsum_float_4x4(acc_0, acc_1, (__m128)__lsx_vldi(0), (__m128)__lsx_vldi(0));

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

#ఒకవేళ defined(__loongarch_asx)
స్థిర స్థిరరాశి int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#అయితే_ముగింపు

శూన్య ggml_vec_dot_iq2_xxs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq2_xxs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K    * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

    స్థిరరాశి uint64_t * signs64 = (స్థిరరాశి uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    స్థిరరాశి uint8_t * aux8 = (స్థిరరాశి uint8_t *)aux32;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి uint16_t * GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            జ్ఞాపకం_నకలు(aux32, q2, 4*పరిమాణం(uint32_t)); q2 += 8;

            స్థిరరాశి __m256i q2_1 = lasx_సమితి_d(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            స్థిరరాశి __m256i q2_2 = lasx_సమితి_d(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            స్థిరరాశి __m256i s2_1 = lasx_సమితి_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            స్థిరరాశి __m256i s2_2 = lasx_సమితి_d(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            స్థిరరాశి __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
            స్థిరరాశి __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
            స్థిరరాశి __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            స్థిరరాశి __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            స్థిరరాశి uint16_t ls1 = aux32[1] >> 28;
            స్థిరరాశి uint16_t ls2 = aux32[3] >> 28;
            స్థిరరాశి __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            స్థిరరాశి __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq2_xs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq2_xs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K   * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

    స్థిరరాశి __m256i mone = __lasx_xvreplgr2vr_b(1);
    స్థిర స్థిరరాశి అక్షరం ఖండం_sign_shuffle_ముసుగు_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    స్థిర స్థిరరాశి అక్షరం ఖండం_sign_shuffle_ముసుగు_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    స్థిర స్థిరరాశి uint8_t బిట్టు_selector_ముసుగు_బైట్లు[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    స్థిరరాశి __m256i బిట్టు_selector_ముసుగు = __lasx_xvld((స్థిరరాశి __m256i*)బిట్టు_selector_ముసుగు_బైట్లు, 0);
    స్థిరరాశి __m256i ఖండం_sign_shuffle_1 = __lasx_xvld((స్థిరరాశి __m256i*)ఖండం_sign_shuffle_ముసుగు_1, 0);
    స్థిరరాశి __m256i ఖండం_sign_shuffle_2 = __lasx_xvld((స్థిరరాశి __m256i*)ఖండం_sign_shuffle_ముసుగు_2, 0);

    స్థిర స్థిరరాశి uint8_t k_బిట్టు_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    స్థిరరాశి __m256i బిట్టు_helper = __lasx_xvld((స్థిరరాశి __m256i*)k_బిట్టు_helper, 0);
    స్థిరరాశి __m256i m511 = __lasx_xvreplgr2vr_h(511);
    స్థిరరాశి __m128i m4 = __lsx_vreplgr2vr_b(0xf);
    స్థిరరాశి __m128i m1 = __lsx_vreplgr2vr_b(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    స్థిరరాశి uint16_t * gindex = (స్థిరరాశి uint16_t *)&aux_gindex;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి uint16_t * GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి int8_t   * GGML_RESTRICT q8 = y[i].qs;

        జ్ఞాపకం_నకలు(&aux64, x[i].scales, 8);
        __m128i stmp = __lsx_vreplgr2vr_d(aux64);
        stmp = __lsx_vilvl_b( __lsx_vand_v(__lsx_vsrli_h(stmp, 4), m4), __lsx_vand_v(stmp, m4));
        స్థిరరాశి __m128i scales = __lsx_vadd_b(__lsx_vslli_h(stmp, 1), m1);

        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            స్థిరరాశి __m256i q2_దత్తాంశం = __lasx_xvld((స్థిరరాశి __m256i*)q2, 0);  q2 += 16;
            aux_gindex = __lasx_xvand_v(q2_దత్తాంశం, m511);

            స్థిరరాశి __m256i partial_sign_బిట్లు = __lasx_xvsrli_h(q2_దత్తాంశం, 9);
            స్థిరరాశి __m256i partial_sign_బిట్లు_పై = __lasx_xvsrli_h(q2_దత్తాంశం, 13);
            స్థిరరాశి __m256i partial_sign_బిట్లు_for_counting = __lasx_xvxor_v(partial_sign_బిట్లు, partial_sign_బిట్లు_పై);

            స్థిరరాశి __m256i odd_బిట్లు = lasx_shuffle_b(బిట్టు_helper, partial_sign_బిట్లు_for_counting);
            స్థిరరాశి __m256i నిండిన_sign_బిట్లు = __lasx_xvor_v(partial_sign_బిట్లు, odd_బిట్లు);

            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_3 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_4 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;

            స్థిరరాశి __m256i q2_1 = lasx_సమితి_d(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            స్థిరరాశి __m256i q2_2 = lasx_సమితి_d(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            స్థిరరాశి __m256i q2_3 = lasx_సమితి_d(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            స్థిరరాశి __m256i q2_4 = lasx_సమితి_d(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            స్థిరరాశి __m128i నిండిన_signs_l = lasx_extracti128(నిండిన_sign_బిట్లు, 0);
            స్థిరరాశి __m128i నిండిన_signs_h = lasx_extracti128(నిండిన_sign_బిట్లు, 1);
            స్థిరరాశి __m256i నిండిన_signs_1 = lasx_insertf128(నిండిన_signs_l, నిండిన_signs_l);
            స్థిరరాశి __m256i నిండిన_signs_2 = lasx_insertf128(నిండిన_signs_h, నిండిన_signs_h);

            __m256i signs;
            signs = lasx_shuffle_b(నిండిన_signs_1, ఖండం_sign_shuffle_1);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, బిట్టు_selector_ముసుగు), బిట్టు_selector_ముసుగు);
            స్థిరరాశి __m256i q8s_1 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_1);

            signs = lasx_shuffle_b(నిండిన_signs_1, ఖండం_sign_shuffle_2);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, బిట్టు_selector_ముసుగు), బిట్టు_selector_ముసుగు);
            స్థిరరాశి __m256i q8s_2 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_2);

            signs = lasx_shuffle_b(నిండిన_signs_2, ఖండం_sign_shuffle_1);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, బిట్టు_selector_ముసుగు), బిట్టు_selector_ముసుగు);
            స్థిరరాశి __m256i q8s_3 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_3);

            signs = lasx_shuffle_b(నిండిన_signs_2, ఖండం_sign_shuffle_2);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, బిట్టు_selector_ముసుగు), బిట్టు_selector_ముసుగు);
            స్థిరరాశి __m256i q8s_4 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_4);

            స్థిరరాశి __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            స్థిరరాశి __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            స్థిరరాశి __m256i dot3  = lasx_maddubs_h(q2_3, q8s_3);
            స్థిరరాశి __m256i dot4  = lasx_maddubs_h(q2_4, q8s_4);

            స్థిరరాశి __m256i sc1 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+0)));
            స్థిరరాశి __m256i sc2 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+1)));
            స్థిరరాశి __m256i sc3 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+2)));
            స్థిరరాశి __m256i sc4 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+3)));

            sumi1 = __lasx_xvadd_w(sumi1, lasx_madd_h(dot1, sc1));
            sumi2 = __lasx_xvadd_w(sumi2, lasx_madd_h(dot2, sc2));
            sumi1 = __lasx_xvadd_w(sumi1, lasx_madd_h(dot3, sc3));
            sumi2 = __lasx_xvadd_w(sumi2, lasx_madd_h(dot4, sc4));
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq2_s_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq2_s * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K  * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

   స్థిర స్థిరరాశి uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    స్థిర స్థిరరాశి uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };


    స్థిరరాశి __m128i m4 = __lsx_vreplgr2vr_b(0xf);
    స్థిరరాశి __m128i m1 = __lsx_vreplgr2vr_b(1);

    స్థిరరాశి __m256i mask1 = __lasx_xvld((స్థిరరాశి __m256i*)k_mask1, 0);
    స్థిరరాశి __m256i mask2 = __lasx_xvld((స్థిరరాశి __m256i*)k_mask2, 0);
    uint64_t aux64;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి uint8_t * GGML_RESTRICT qs = x[i].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి uint16_t * GGML_RESTRICT signs = (స్థిరరాశి uint16_t *)(x[i].qs + QK_K/8);
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        __m128i tmp1;
        జ్ఞాపకం_నకలు(&aux64, x[i].scales, 8);
        tmp1 = __lsx_vinsgr2vr_d(tmp1, aux64, 0);
        tmp1 = __lsx_vinsgr2vr_d(tmp1, aux64 >> 4, 1);
        స్థిరరాశి __m128i scales8 = __lsx_vadd_b(__lsx_vslli_h(__lsx_vand_v(tmp1, m4), 1), m1);
        స్థిరరాశి __m256i scales16 = lasx_ext8_16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q2_1 = lasx_సమితి_d(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            స్థిరరాశి __m256i q2_2 = lasx_సమితి_d(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = __lasx_xvreplgr2vr_w(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            స్థిరరాశి __m256i s2_1 = __lasx_xvseq_b(aux256, mask2);
            స్థిరరాశి __m256i q8s_1 = __lasx_xvsub_b(__lasx_xvxor_v(s2_1, q8_1), s2_1);

            aux256 = __lasx_xvreplgr2vr_w(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            స్థిరరాశి __m256i s2_2 = __lasx_xvseq_b(aux256, mask2);
            స్థిరరాశి __m256i q8s_2 = __lasx_xvsub_b(__lasx_xvxor_v(s2_2, q8_2), s2_2);

            signs += 4;

            స్థిరరాశి __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            స్థిరరాశి __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            స్థిరరాశి __m256i p1 = lasx_madd_h(dot1, lasx_shuffle_b(scales16, get_scale_shuffle_k4(ib32+0)));
            స్థిరరాశి __m256i p2 = lasx_madd_h(dot2, lasx_shuffle_b(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq3_xxs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq3_xxs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K    * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

    స్థిరరాశి uint64_t * signs64 = (స్థిరరాశి uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = (__m256)__lasx_xvldi(0);
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి uint8_t * GGML_RESTRICT q3 = x[i].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q2_1 = lasx_సమితి_w(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            స్థిరరాశి __m256i q2_2 = lasx_సమితి_w(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            జ్ఞాపకం_నకలు(aux32, gas, 8); gas += 8;

            స్థిరరాశి __m256i s2_1 = lasx_సమితి_d(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            స్థిరరాశి __m256i s2_2 = lasx_సమితి_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            స్థిరరాశి __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
            స్థిరరాశి __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
            స్థిరరాశి __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            స్థిరరాశి __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            స్థిరరాశి uint16_t ls1 = aux32[0] >> 28;
            స్థిరరాశి uint16_t ls2 = aux32[1] >> 28;

            స్థిరరాశి __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            స్థిరరాశి __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.25f * hsum_float_8(accumf);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq3_s_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq3_s * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K  * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

   స్థిర స్థిరరాశి uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    స్థిర స్థిరరాశి uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    స్థిరరాశి __m256i mask1 = __lasx_xvld((స్థిరరాశి __m256i*)k_mask1, 0);
    స్థిరరాశి __m256i mask2 = __lasx_xvld((స్థిరరాశి __m256i*)k_mask2, 0);

    __m256i idx_జరుపు = lasx_సమితి_w(1, 2, 3, 4, 5, 6, 7, 8);
    స్థిరరాశి __m256i idx_ముసుగు  = __lasx_xvreplgr2vr_w(256);

    రకనిర్వచనం సంఘం {
        __m256i  vec[2];
        uint32_t సూచీ[16];
    } సూచీ_t;

    సూచీ_t idx;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి uint8_t * GGML_RESTRICT qs = x[i].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి uint16_t * GGML_RESTRICT signs = (స్థిరరాశి uint16_t *)x[i].signs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            స్థిరరాశి __m256i q8_1 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8_2 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i idx_l = lasx_extu8_16(__lsx_vld(qs, 0)); qs += 16;
            idx.vec[0] = __lasx_xvreplgr2vr_w(qh[ib32+0]);
            idx.vec[1] = __lasx_xvreplgr2vr_w(qh[ib32+1]);
            idx.vec[0] = __lasx_xvand_v(__lasx_xvsll_w(idx.vec[0], idx_జరుపు), idx_ముసుగు);
            idx.vec[1] = __lasx_xvand_v(__lasx_xvsll_w(idx.vec[1], idx_జరుపు), idx_ముసుగు);
            idx.vec[0] = __lasx_xvor_v(idx.vec[0], lasx_ext16_32(lasx_extracti128(idx_l, 0)));
            idx.vec[1] = __lasx_xvor_v(idx.vec[1], lasx_ext16_32(lasx_extracti128(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            స్థిరరాశి __m256i q2_1 = lasx_సమితి_w(
                    iq3s_grid[idx.సూచీ[7]], iq3s_grid[idx.సూచీ[6]], iq3s_grid[idx.సూచీ[5]], iq3s_grid[idx.సూచీ[4]],
                    iq3s_grid[idx.సూచీ[3]], iq3s_grid[idx.సూచీ[2]], iq3s_grid[idx.సూచీ[1]], iq3s_grid[idx.సూచీ[0]]
            );
            స్థిరరాశి __m256i q2_2 = lasx_సమితి_w(
                    iq3s_grid[idx.సూచీ[15]], iq3s_grid[idx.సూచీ[14]], iq3s_grid[idx.సూచీ[13]], iq3s_grid[idx.సూచీ[12]],
                    iq3s_grid[idx.సూచీ[11]], iq3s_grid[idx.సూచీ[10]], iq3s_grid[idx.సూచీ[ 9]], iq3s_grid[idx.సూచీ[ 8]]
            );

            __m256i aux256 = __lasx_xvreplgr2vr_w(signs[0] | (signs[1] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            స్థిరరాశి __m256i s2_1 = __lasx_xvseq_b(aux256, mask2);
            స్థిరరాశి __m256i q8s_1 = __lasx_xvsub_b(__lasx_xvxor_v(s2_1, q8_1), s2_1);

            aux256 = __lasx_xvreplgr2vr_w(signs[2] | (signs[3] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            స్థిరరాశి __m256i s2_2 = __lasx_xvseq_b(aux256, mask2);
            స్థిరరాశి __m256i q8s_2 = __lasx_xvsub_b(__lasx_xvxor_v(s2_2, q8_2), s2_2);

            signs += 4;

            స్థిరరాశి __m256i dot1 = lasx_maddubs_h(q2_1, q8s_1);
            స్థిరరాశి __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            స్థిరరాశి uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            స్థిరరాశి uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            స్థిరరాశి __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            స్థిరరాశి __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = hsum_float_8(accumf);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

#ఒకవేళ defined(__loongarch_asx)
స్థిర అంతఃవరుస __m256i mul_కూడు_epi8(స్థిరరాశి __m256i x, స్థిరరాశి __m256i y) {
    స్థిరరాశి __m256i a = __lasx_xvmulwev_h_b(x, y);
    స్థిరరాశి __m256i b = __lasx_xvmulwod_h_b(x, y);
    తిరిగి __lasx_xvadd_h(a, b);
}
#అయితే_ముగింపు

శూన్య ggml_vec_dot_iq1_s_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq1_s * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K  * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

    __m256 accum = (__m256)__lasx_xvldi(0);
    దశాంశం accum1 = 0;
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {

        స్థిరరాశి int8_t   * q8 = y[i].qs;
        స్థిరరాశి uint8_t  * qs = x[i].qs;
        స్థిరరాశి uint16_t * qh = x[i].qh;

        __m256i sumi = __lasx_xvldi(0);
        పూర్ణాంకం sumi1 = 0;
        కొరకు (పూర్ణాంకం ib = 0; ib < QK_K/32; ib += 2) {
            __m256i q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)], 0);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], 1);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)], 2);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], 3);

            __m256i q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)], 0);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], 1);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)], 2);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], 3);

            qs += 8;
            స్థిరరాశి __m256i q8b_1 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8b_2 = __lasx_xvld((స్థిరరాశి __m256i*)q8, 0); q8 += 32;

            స్థిరరాశి __m256i dot1 = mul_కూడు_epi8(q1b_1, q8b_1);
            స్థిరరాశి __m256i dot2 = mul_కూడు_epi8(q1b_2, q8b_2);
            స్థిరరాశి int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            స్థిరరాశి int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;

            __m256i tmp1, tmp5, tmp6;
            tmp1 = __lasx_xvreplgr2vr_h(ls1);
            tmp5 = __lasx_xvmulwev_w_h(dot1, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(dot1, tmp1);
            స్థిరరాశి __m256i p1 = __lasx_xvadd_w(tmp5, tmp6);

            tmp1 = __lasx_xvreplgr2vr_h(ls2);
            tmp5 = __lasx_xvmulwev_w_h(dot2, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(dot2, tmp1);
            స్థిరరాశి __m256i p2 = __lasx_xvadd_w(tmp5, tmp6);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        accum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), accum);
        accum1 += d * sumi1;
    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq4_nl_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);
    ప్రతిజ్ఞా(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    స్థిరరాశి ఖండం_iq4_nl * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0   * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK4_NL;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined (__loongarch_asx)

    స్థిరరాశి __m128i values128 = __lsx_vld((స్థిరరాశి __m128i*)kvalues_iq4nl, 0);
    స్థిరరాశి __m128i m4b  = __lsx_vreplgr2vr_b(0x0f);
    స్థిరరాశి __m256i mone = __lasx_xvreplgr2vr_h(1);

    __m256 accum1 = (__m256)__lasx_xvldi(0);
    __m256 accum2 = (__m256)__lasx_xvldi(0);
    కొరకు (; ib + 1 < nb; ib += 2) {
        స్థిరరాశి __m128i q4bits_1 = __lsx_vld((స్థిరరాశి __m128i*)x[ib + 0].qs, 0);
        స్థిరరాశి __m128i q4bits_2 = __lsx_vld((స్థిరరాశి __m128i*)x[ib + 1].qs, 0);
        స్థిరరాశి __m256i q8b_1 = __lasx_xvld((స్థిరరాశి __m256i *)y[ib + 0].qs, 0);
        స్థిరరాశి __m256i q8b_2 = __lasx_xvld((స్థిరరాశి __m256i *)y[ib + 1].qs, 0);
        స్థిరరాశి __m256i q4b_1 = lasx_insertf128(lsx_shuffle_b(values128, __lsx_vand_v(__lsx_vsrli_h(q4bits_1, 4), m4b)),
                                              lsx_shuffle_b(values128, __lsx_vand_v(q4bits_1, m4b)));
        స్థిరరాశి __m256i q4b_2 = lasx_insertf128(lsx_shuffle_b(values128, __lsx_vand_v(__lsx_vsrli_h(q4bits_2, 4), m4b)),
                                              lsx_shuffle_b(values128, __lsx_vand_v(q4bits_2, m4b)));
        స్థిరరాశి __m256i p16_1 = mul_కూడు_epi8(q4b_1, q8b_1);
        స్థిరరాశి __m256i p16_2 = mul_కూడు_epi8(q4b_2, q8b_2);
        స్థిరరాశి __m256i p_1 = lasx_madd_h(p16_1, mone);
        స్థిరరాశి __m256i p_2 = lasx_madd_h(p16_2, mone);
        accum1 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(y[ib + 0].d)*GGML_CPU_FP16_గమ్యం_FP32(x[ib + 0].d)),
                __lasx_xvffint_s_w(p_1), accum1);
        accum2 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(y[ib + 1].d)*GGML_CPU_FP16_గమ్యం_FP32(x[ib + 1].d)),
                __lasx_xvffint_s_w(p_2), accum2);
    }

    sumf = hsum_float_8(__lasx_xvfadd_s(accum1, accum2));

#అయితే_ముగింపు
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(y[ib].d)*GGML_CPU_FP16_గమ్యం_FP32(x[ib].d);
        పూర్ణాంకం sumi1 = 0, sumi2 = 0;
        కొరకు (పూర్ణాంకం j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

శూన్య ggml_vec_dot_iq4_xs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);
    ప్రతిజ్ఞా(n % QK_K == 0);

    స్థిరరాశి ఖండం_iq4_xs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K   * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__loongarch_asx)

    స్థిరరాశి __m128i values128 = __lsx_vld((స్థిరరాశి __m128i*)kvalues_iq4nl, 0);

    __m256 accum = (__m256)__lasx_xvldi(0);

    కొరకు (పూర్ణాంకం ibl = 0; ibl < nb; ++ibl) {
        స్థిరరాశి uint8_t * qs = x[ibl].qs;
        స్థిరరాశి int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        కొరకు (పూర్ణాంకం ib = 0; ib < QK_K/32; ib += 2) {
            స్థిరరాశి __m128i q4bits_1 = __lsx_vld((స్థిరరాశి __m128i*)qs, 0); qs += 16;
            స్థిరరాశి __m128i q4bits_2 = __lsx_vld((స్థిరరాశి __m128i*)qs, 0); qs += 16;
            స్థిరరాశి __m256i q8b_1 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q8b_2 = __lasx_xvld((స్థిరరాశి __m256i *)q8, 0); q8 += 32;
            స్థిరరాశి __m256i q4b_1 = lasx_insertf128(__lsx_vshuf_b(values128, values128, __lsx_vsrli_b(q4bits_1, 4)),
                                                  __lsx_vshuf_b(values128, values128, __lsx_vandi_b(q4bits_1, 0xf)));
            స్థిరరాశి __m256i q4b_2 = lasx_insertf128(__lsx_vshuf_b(values128, values128, __lsx_vsrli_b(q4bits_2, 4)),
                                                  __lsx_vshuf_b(values128, values128, __lsx_vandi_b(q4bits_2, 0xf)));
            స్థిరరాశి __m256i p16_1 = mul_కూడు_epi8(q4b_1, q8b_1);
            స్థిరరాశి __m256i p16_2 = mul_కూడు_epi8(q4b_2, q8b_2);
            స్థిరరాశి int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            స్థిరరాశి int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            స్థిరరాశి __m256i p_1 = lasx_madd_h(p16_1, __lasx_xvreplgr2vr_h(ls1));
            స్థిరరాశి __m256i p_2 = lasx_madd_h(p16_2, __lasx_xvreplgr2vr_h(ls2));
            sumi1 = __lasx_xvadd_w(p_1, sumi1);
            sumi2 = __lasx_xvadd_w(p_2, sumi2);
        }
        accum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_CPU_FP16_గమ్యం_FP32(x[ibl].d)*y[ibl].d),
                __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);

#లేదావేళ defined(__loongarch_sx)

    స్థిరరాశి __m128i values128 = __lsx_vld((స్థిరరాశి __m128i*)kvalues_iq4nl, 0);

    __m128 accum = (__m128)__lsx_vldi(0);
    కొరకు (పూర్ణాంకం ibl = 0; ibl < nb; ++ibl) {
        స్థిరరాశి uint8_t * qs = x[ibl].qs;
        స్థిరరాశి int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m128i sumi = __lsx_vldi(0);
        కొరకు (పూర్ణాంకం ib = 0; ib < QK_K/32; ++ib) {
            స్థిరరాశి __m128i q4bits = __lsx_vld((స్థిరరాశి __m128i*)qs, 0); qs += 16;
            స్థిరరాశి __m128i q8b_0 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q8b_1 = __lsx_vld((స్థిరరాశి __m128i*)q8, 0); q8 += 16;
            స్థిరరాశి __m128i q4b_0 = __lsx_vshuf_b(values128, values128, __lsx_vandi_b(q4bits, 0xf));
            స్థిరరాశి __m128i q4b_1 = __lsx_vshuf_b(values128, values128, __lsx_vsrli_b(q4bits, 4));
            స్థిరరాశి __m128i p16_0 = lsx_maddubs_h(q4b_0, q8b_0);
            స్థిరరాశి __m128i p16_1 = lsx_maddubs_h(q4b_1, q8b_1);
            స్థిరరాశి int16_t ls = (((x[ibl].scales_l[ib/2] >> ((ib & 1) * 4)) & 0xf) | ((sh & 0x3) << 4)) - 32;
            sh >>= 2;
            sumi = __lsx_vadd_w(lsx_madd_h(p16_0, __lsx_vreplgr2vr_h(ls)), sumi);
            sumi = __lsx_vadd_w(lsx_madd_h(p16_1, __lsx_vreplgr2vr_h(ls)), sumi);
        }
        స్థిరరాశి దశాంశం ds = GGML_CPU_FP16_గమ్యం_FP32(x[ibl].d) * y[ibl].d;
        accum = __lsx_vfadd_s(__lsx_vfmul_s(__lsx_vreplfr2vr_s(ds), __lsx_vffint_s_w(sumi)), accum);
    }

    *s = ((v4f32)lsx_hadd_s(lsx_hadd_s(accum, accum), lsx_hadd_s(accum, accum)))[0];

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}
