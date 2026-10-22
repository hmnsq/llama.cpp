#ifndef GGML_SYCL_ESIMD_HPP
#define GGML_SYCL_ESIMD_HPP

#অন্তর্ভাব <sycl/ext/intel/esimd.hpp>

#অন্তর্ভাব "common.hpp"

নামাকাশ ggml_sycl_esimd {

constexpr পূর্ণসংখ্যা GGML_SYCL_DMMV_ESIMD_WG_আকার = 4;

//
// Shared ESIMD building blocks for the reordered K-quant dequantize-matvec
// kernels.
//
// The reordered K-quant ESIMD matvec kernels share one skeleton: per super-block,
// load a 256-float activation slice, load one weight block, dequantize it into 8
// chunks of 32 and MAC each chunk against the matching activation slice, then
// reduce and run a lane-0 epilogue.
//
// Each K-quant kernel emits exactly 8 chunks of 32 mapping to activation slices
// 0..7, so the per-block work is captured by esimd_reorder_q_traits<T>::mac_pair,
// which dequantizes two weight blocks and MACs both against a shared activation
// vector with the two FMA chains interleaved (co-scheduled to hide FMA latency).
// The "pair" is the (row0,row1) row pair owned by one work-group, so the
// layout+dequant is written once per quant type here.
//

template <ggml_type T> কাঠামো esimd_reorder_q_traits;

// build a 32-lane vector whose low 16 lanes are `lo` and high 16 are `hi`
// (a super-chunk splits into two 16-wide halves with distinct scale/min codes).
স্থির ESIMD_INLINE sycl::ext::intel::esimd::simd<দশমিক, 32> splat_lo_hi(দশমিক lo, দশমিক hi) {
    using নামাকাশ sycl::ext::intel::esimd;
    simd<দশমিক, 32> v;
    v.select<16, 1>(0)  = lo;
    v.select<16, 1>(16) = hi;
    ফেরত v;
}

// unpack one block of Q4_K/Q5_K scale/min codes (get_scale_min_k4 layout) into 8
// float scales (dall * sc) and 8 float mins (-dmin * m); the min carries the
// negation so the dequant epilogue adds.
স্থির ESIMD_INLINE শূন্য unpack_scale_min_k4(
        sycl::ext::intel::esimd::simd<uint8_t, 12> scales, দশমিক dall, দশমিক dmin,
        sycl::ext::intel::esimd::simd<দশমিক, 8> & scale_f,
        sycl::ext::intel::esimd::simd<দশমিক, 8> & min_f) {
    using নামাকাশ sycl::ext::intel::esimd;
    simd<uint8_t, 8> sc = 0;
    simd<uint8_t, 8> m  = 0;
    simd<uint8_t, 4> scale_lo = scales.select<4, 1>(0);
    simd<uint8_t, 4> min_lo   = scales.select<4, 1>(4);
    simd<uint8_t, 4> hi_bits  = scales.select<4, 1>(8);
    sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
    sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                         ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
    m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                         ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    scale_f = convert<দশমিক>(sc) * dall;
    min_f   = convert<দশমিক>(m) * (-dmin);
}

// ---------------------------------------------------------------------------
// Q2_K, SOA reorder layout produced by reorder_qw_q2_k:
//   [qs: nb*(QK_K/4)] [scales: nb*(QK_K/16)] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// 2 bits per weight. The 8 output chunks of 32 (matching dequantize_row_q2_K)
// map to super-chunk s (0..7): byte base 32*(s/4) into the 64-byte qs array,
// bit shift 2*(s%4); the low 16 lanes use scales[2s], the high 16 use
// scales[2s+1], with dl = d*(sc & 0xF), ml = dmin*(sc >> 4), deq = dl*q - ml.
// ---------------------------------------------------------------------------
template <> কাঠামো esimd_reorder_q_traits<GGML_TYPE_Q2_K> {
    কাঠামো ptrs {
        ধ্রুবক uint8_t *    qs;
        ধ্রুবক uint8_t *    scales;
        ধ্রুবক sycl::half * dm;
    };

    স্থির ESIMD_INLINE ptrs make_ptrs(ধ্রুবক শূন্য * vx, size_t nb) {
        ধ্রুবক uint8_t * qs     = (ধ্রুবক uint8_t *) vx;
        ধ্রুবক uint8_t * scales = qs + nb * (QK_K / 4);
        ধ্রুবক sycl::half * dm  = (ধ্রুবক sycl::half *) (scales + nb * (QK_K / 16));
        ফেরত { qs, scales, dm };
    }

    স্থির ESIMD_INLINE শূন্য mac_pair(
            ধ্রুবক ptrs & pa, size_t bia,
            ধ্রুবক ptrs & pb, size_t bib, যৌক্তিক has_b,
            sycl::ext::intel::esimd::simd<দশমিক, 256> & y_vec,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_a,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_b) {
        using নামাকাশ sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 16> scales_a = block_load<uint8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<uint8_t, 16> scales_b = 0;

        ধ্রুবক দশমিক dall_a = (দশমিক) pa.dm[bia * 2 + 0];
        ধ্রুবক দশমিক dmin_a = (দশমিক) pa.dm[bia * 2 + 1];
        দশমিক dall_b = 0.0f;
        দশমিক dmin_b = 0.0f;
        যদি (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            scales_b = block_load<uint8_t, 16>(pb.scales + bib * (QK_K / 16));
            dall_b = (দশমিক) pb.dm[bib * 2 + 0];
            dmin_b = (দশমিক) pb.dm[bib * 2 + 1];
        }

        // per-chunk scale (d * (sc & 0xF)) and min (-dmin * (sc >> 4)), all 16 codes;
        // min carries the negation so the dequant epilogue adds (matches Q4_K/Q5_K)
        simd<দশমিক, 16> scale_f_a = convert<দশমিক>(scales_a & simd<uint8_t, 16>(0x0F)) * dall_a;
        simd<দশমিক, 16> min_f_a   = convert<দশমিক>(scales_a >> simd<uint8_t, 16>(4))  * (-dmin_a);
        simd<দশমিক, 16> scale_f_b = convert<দশমিক>(scales_b & simd<uint8_t, 16>(0x0F)) * dall_b;
        simd<দশমিক, 16> min_f_b   = convert<দশমিক>(scales_b >> simd<uint8_t, 16>(4))  * (-dmin_b);

#pragma unroll
        জন্য (পূর্ণসংখ্যা s = 0; s < 8; ++s) {
            ধ্রুবক পূর্ণসংখ্যা     byte_base = 32 * (s / 4);
            ধ্রুবক uint8_t shift     = (uint8_t) (2 * (s % 4));
            simd<দশমিক, 32> y_s = y_vec.select<32, 1>(s * 32);

            simd<uint8_t, 32> qa = (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);
            simd<uint8_t, 32> qb = (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3);

            ধ্রুবক দশমিক scale_a_lo = scale_f_a[2 * s + 0];
            ধ্রুবক দশমিক scale_a_hi = scale_f_a[2 * s + 1];
            ধ্রুবক দশমিক min_a_lo   = min_f_a[2 * s + 0];
            ধ্রুবক দশমিক min_a_hi   = min_f_a[2 * s + 1];
            ধ্রুবক দশমিক scale_b_lo = scale_f_b[2 * s + 0];
            ধ্রুবক দশমিক scale_b_hi = scale_f_b[2 * s + 1];
            ধ্রুবক দশমিক min_b_lo   = min_f_b[2 * s + 0];
            ধ্রুবক দশমিক min_b_hi   = min_f_b[2 * s + 1];

            simd<দশমিক, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<দশমিক, 32> min_vec_a   = splat_lo_hi(min_a_lo, min_a_hi);
            simd<দশমিক, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);
            simd<দশমিক, 32> min_vec_b   = splat_lo_hi(min_b_lo, min_b_hi);

            simd<দশমিক, 32> deq_a = convert<দশমিক>(qa) * scale_vec_a + min_vec_a;
            simd<দশমিক, 32> deq_b = convert<দশমিক>(qb) * scale_vec_b + min_vec_b;

            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

// ---------------------------------------------------------------------------
// Q3_K, SOA reorder layout produced by reorder_qw_q3_k:
//   [qs: nb*(QK_K/4)] [hmask: nb*(QK_K/8)] [scales: nb*12] [d: nb*sizeof(half)]
// with nb = nrows*num_blocks_per_row. Single super-block scale d, no dmin.
//
// 3 bits per weight: 2 low bits in qs, 1 high bit in hmask. The 8 output chunks
// of 32 (matching dequantize_row_q3_K) map to super-chunk s (0..7): byte base
// 32*(s/4) into the 64-byte qs array, bit shift 2*(s%4); the low 16 lanes use
// scale code 2s, the high 16 use 2s+1. hmask is a 32-byte array (like Q5_K's
// qh) where chunk s uses bit s of the same 32 bytes, but INVERTED: the value is
// (q & 3) - (hmask_bit_set ? 0 : 4), i.e. (q & 3) + 4*bit - 4.
//
// The 16 6-bit scale codes are packed into 12 bytes (get_scale_min layout for
// Q3_K): low nibbles from bytes 0..7, high 2 bits from bytes 8..11 shifted by
// 0/2/4/6; the dequant scale is d * (code - 32).
// ---------------------------------------------------------------------------
template <> কাঠামো esimd_reorder_q_traits<GGML_TYPE_Q3_K> {
    কাঠামো ptrs {
        ধ্রুবক uint8_t *    qs;
        ধ্রুবক uint8_t *    hmask;
        ধ্রুবক uint8_t *    scales;
        ধ্রুবক sycl::half * d;
    };

    স্থির ESIMD_INLINE ptrs make_ptrs(ধ্রুবক শূন্য * vx, size_t nb) {
        ধ্রুবক uint8_t * qs     = (ধ্রুবক uint8_t *) vx;
        ধ্রুবক uint8_t * hmask  = qs + nb * (QK_K / 4);
        ধ্রুবক uint8_t * scales = hmask + nb * (QK_K / 8);
        ধ্রুবক sycl::half * d   = (ধ্রুবক sycl::half *) (scales + nb * 12);
        ফেরত { qs, hmask, scales, d };
    }

    // unpack the 12 packed bytes into 16 6-bit scale codes (dequantize_row_q3_K
    // aux layout), returned as float scale = d * (code - 32).
    // done with wide (8/16-lane) ops rather than four 4-lane groups.
    স্থির ESIMD_INLINE sycl::ext::intel::esimd::simd<দশমিক, 16> unpack_scales(
            sycl::ext::intel::esimd::simd<uint8_t, 12> in, দশমিক d) {
        using নামাকাশ sycl::ext::intel::esimd;

        // low 6-bit part: codes 0..7 = low nibble of bytes 0..7,
        //                 codes 8..15 = high nibble of bytes 0..7
        simd<uint8_t, 8>  lo8 = in.select<8, 1>(0);
        simd<uint8_t, 16> code;
        code.select<8, 1>(0) = lo8 & simd<uint8_t, 8>(0x0F);
        code.select<8, 1>(8) = lo8 >> simd<uint8_t, 8>(4);

        // high 2-bit part: bytes 8..11 replicated 4x, group g (0..3) shifted 2*g
        simd<uint8_t, 16> hib;
        hib.select<4, 1>(0)  = in.select<4, 1>(8);
        hib.select<4, 1>(4)  = in.select<4, 1>(8);
        hib.select<4, 1>(8)  = in.select<4, 1>(8);
        hib.select<4, 1>(12) = in.select<4, 1>(8);
        simd<uint8_t, 16> hshift;
        hshift.select<4, 1>(0)  = 0;
        hshift.select<4, 1>(4)  = 2;
        hshift.select<4, 1>(8)  = 4;
        hshift.select<4, 1>(12) = 6;
        hib = (hib >> hshift) & simd<uint8_t, 16>(0x03);

        code = code | (hib << simd<uint8_t, 16>(4));
        ফেরত (convert<দশমিক>(code) - 32.0f) * d;
    }

    স্থির ESIMD_INLINE শূন্য mac_pair(
            ধ্রুবক ptrs & pa, size_t bia,
            ধ্রুবক ptrs & pb, size_t bib, যৌক্তিক has_b,
            sycl::ext::intel::esimd::simd<দশমিক, 256> & y_vec,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_a,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_b) {
        using নামাকাশ sycl::ext::intel::esimd;

        simd<uint8_t, 64> qs_a     = block_load<uint8_t, 64>(pa.qs + bia * (QK_K / 4));
        simd<uint8_t, 64> qs_b     = 0;
        simd<uint8_t, 32> hmask_a  = block_load<uint8_t, 32>(pa.hmask + bia * (QK_K / 8));
        simd<uint8_t, 32> hmask_b  = 0;
        simd<uint8_t, 12> scales_a = block_load<uint8_t, 12>(pa.scales + bia * 12);
        simd<uint8_t, 12> scales_b = 0;

        ধ্রুবক দশমিক d_a = (দশমিক) pa.d[bia];
        দশমিক d_b = 0.0f;
        যদি (has_b) {
            qs_b     = block_load<uint8_t, 64>(pb.qs + bib * (QK_K / 4));
            hmask_b  = block_load<uint8_t, 32>(pb.hmask + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * 12);
            d_b = (দশমিক) pb.d[bib];
        }

        simd<দশমিক, 16> scale_f_a = unpack_scales(scales_a, d_a);
        simd<দশমিক, 16> scale_f_b = unpack_scales(scales_b, d_b);

#pragma unroll
        জন্য (পূর্ণসংখ্যা s = 0; s < 8; ++s) {
            ধ্রুবক পূর্ণসংখ্যা     byte_base = 32 * (s / 4);
            ধ্রুবক uint8_t shift     = (uint8_t) (2 * (s % 4));
            simd<দশমিক, 32> y_s = y_vec.select<32, 1>(s * 32);

            // 2 low bits from qs, high bit from hmask (bit s of the same 32 bytes);
            // value = (q & 3) + 4*bit - 4  (inverted hmask: subtract 4 when bit clear).
            // merge in the integer domain: q3 = (q & 3) | (bit << 2) in {0..7},
            // then a single convert + subtract yields q3 - 4 (one convert, not two)
            simd<uint16_t, 32> q3_a = convert<uint16_t>(
                    (qs_a.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_a |= convert<uint16_t>(
                    ((hmask_a >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));
            simd<uint16_t, 32> q3_b = convert<uint16_t>(
                    (qs_b.select<32, 1>(byte_base) >> shift) & simd<uint8_t, 32>(3));
            q3_b |= convert<uint16_t>(
                    ((hmask_b >> simd<uint8_t, 32>((uint8_t) s)) & simd<uint8_t, 32>(1)) << simd<uint8_t, 32>(2));

            simd<দশমিক, 32> qf_a = convert<দশমিক>(q3_a) - 4.0f;
            simd<দশমিক, 32> qf_b = convert<দশমিক>(q3_b) - 4.0f;

            ধ্রুবক দশমিক scale_a_lo = scale_f_a[2 * s + 0];
            ধ্রুবক দশমিক scale_a_hi = scale_f_a[2 * s + 1];
            ধ্রুবক দশমিক scale_b_lo = scale_f_b[2 * s + 0];
            ধ্রুবক দশমিক scale_b_hi = scale_f_b[2 * s + 1];

            simd<দশমিক, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
            simd<দশমিক, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

            simd<দশমিক, 32> deq_a = qf_a * scale_vec_a;
            simd<দশমিক, 32> deq_b = qf_b * scale_vec_b;

            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

// ---------------------------------------------------------------------------
// Q4_K, SOA reorder layout produced by reorder_qw_q4_k:
//   [qs: nb*(QK_K/2)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
// ---------------------------------------------------------------------------
template <> কাঠামো esimd_reorder_q_traits<GGML_TYPE_Q4_K> {
    কাঠামো ptrs {
        ধ্রুবক uint8_t *    qs;
        ধ্রুবক uint8_t *    scales;
        ধ্রুবক sycl::half * dm;
    };

    স্থির ESIMD_INLINE ptrs make_ptrs(ধ্রুবক শূন্য * vx, size_t nb) {
        ধ্রুবক uint8_t * qs     = (ধ্রুবক uint8_t *) vx;
        ধ্রুবক uint8_t * scales = qs + nb * (QK_K / 2);
        ধ্রুবক sycl::half * dm  = (ধ্রুবক sycl::half *) (scales + nb * K_SCALE_আকার);
        ফেরত { qs, scales, dm };
    }

    স্থির ESIMD_INLINE শূন্য mac_pair(
            ধ্রুবক ptrs & pa, size_t bia,
            ধ্রুবক ptrs & pb, size_t bib, যৌক্তিক has_b,
            sycl::ext::intel::esimd::simd<দশমিক, 256> & y_vec,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_a,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_b) {
        using নামাকাশ sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_আকার);
        simd<uint8_t, 12>  scales_b = 0;

        ধ্রুবক দশমিক dall_a = (দশমিক) pa.dm[bia * 2 + 0];
        ধ্রুবক দশমিক dmin_a = (দশমিক) pa.dm[bia * 2 + 1];
        দশমিক dall_b = 0.0f;
        দশমিক dmin_b = 0.0f;
        যদি (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_আকার);
            dall_b = (দশমিক) pb.dm[bib * 2 + 0];
            dmin_b = (দশমিক) pb.dm[bib * 2 + 1];
        }

        simd<দশমিক, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb += 2) {
            ধ্রুবক পূর্ণসংখ্যা q_offset = sb * 16;
            simd<দশমিক, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<দশমিক, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            ধ্রুবক দশমিক scale_a_lo = scale_f_a[sb];
            ধ্রুবক দশমিক scale_a_hi = scale_f_a[sb + 1];
            ধ্রুবক দশমিক min_a_lo   = min_f_a[sb];
            ধ্রুবক দশমিক min_a_hi   = min_f_a[sb + 1];
            ধ্রুবক দশমিক scale_b_lo = scale_f_b[sb];
            ধ্রুবক দশমিক scale_b_hi = scale_f_b[sb + 1];
            ধ্রুবক দশমিক min_b_lo   = min_f_b[sb];
            ধ্রুবক দশমিক min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi = qs_hi_b.select<32, 1>(q_offset);

            simd<দশমিক, 32> deq_a_lo = convert<দশমিক>(qa_lo) * scale_a_lo + min_a_lo;
            simd<দশমিক, 32> deq_a_hi = convert<দশমিক>(qa_hi) * scale_a_hi + min_a_hi;
            simd<দশমিক, 32> deq_b_lo = convert<দশমিক>(qb_lo) * scale_b_lo + min_b_lo;
            simd<দশমিক, 32> deq_b_hi = convert<দশমিক>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q5_K, SOA reorder layout produced by reorder_qw_q5_k:
//   [qs: nb*(QK_K/2)] [qh: nb*(QK_K/8)] [scales: nb*K_SCALE_SIZE] [dm: nb*sizeof(half2)]
// with nb = nrows*num_blocks_per_row.
//
// Identical to Q4_K except each 4-bit quant gains a 5th (high) bit from qh:
// output chunk c (0..7) adds 16 when bit c of qh[l] is set, where qh[l] indexes
// the same 32 bytes for every chunk (matches dequantize_row_q5_K).
// ---------------------------------------------------------------------------
template <> কাঠামো esimd_reorder_q_traits<GGML_TYPE_Q5_K> {
    কাঠামো ptrs {
        ধ্রুবক uint8_t *    qs;
        ধ্রুবক uint8_t *    qh;
        ধ্রুবক uint8_t *    scales;
        ধ্রুবক sycl::half * dm;
    };

    স্থির ESIMD_INLINE ptrs make_ptrs(ধ্রুবক শূন্য * vx, size_t nb) {
        ধ্রুবক uint8_t * qs     = (ধ্রুবক uint8_t *) vx;
        ধ্রুবক uint8_t * qh     = qs + nb * (QK_K / 2);
        ধ্রুবক uint8_t * scales = qh + nb * (QK_K / 8);
        ধ্রুবক sycl::half * dm  = (ধ্রুবক sycl::half *) (scales + nb * K_SCALE_আকার);
        ফেরত { qs, qh, scales, dm };
    }

    // extract bit `bit` (0..7) of each lane and move it to bit position 4,
    // e.g. for the 4-bit base quant's 5th (high) bit. `bit` is always a
    // compile-time-known unrolled loop constant at call sites, so this folds
    // to a single mask (bit==4), mask+left-shift (bit<4), or mask+right-shift
    // (bit>4) instead of the shift+mask+shift a naive `(qh>>bit & 1) << 4` emits.
    স্থির ESIMD_INLINE sycl::ext::intel::esimd::simd<uint16_t, 32> extract_bit_to_pos4(
            sycl::ext::intel::esimd::simd<uint8_t, 32> qh, পূর্ণসংখ্যা bit) {
        using নামাকাশ sycl::ext::intel::esimd;
        simd<uint16_t, 32> masked = convert<uint16_t>(qh & simd<uint8_t, 32>((uint8_t) (1u << bit)));
        যদি (bit < 4) {
            ফেরত masked << simd<uint16_t, 32>((uint16_t) (4 - bit));
        } অন্যথা যদি (bit > 4) {
            ফেরত masked >> simd<uint16_t, 32>((uint16_t) (bit - 4));
        }
        ফেরত masked;
    }

    স্থির ESIMD_INLINE শূন্য mac_pair(
            ধ্রুবক ptrs & pa, size_t bia,
            ধ্রুবক ptrs & pb, size_t bib, যৌক্তিক has_b,
            sycl::ext::intel::esimd::simd<দশমিক, 256> & y_vec,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_a,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_b) {
        using নামাকাশ sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * (QK_K / 2));
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 32>  qh_a     = block_load<uint8_t, 32>(pa.qh + bia * (QK_K / 8));
        simd<uint8_t, 32>  qh_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * K_SCALE_আকার);
        simd<uint8_t, 12>  scales_b = 0;

        ধ্রুবক দশমিক dall_a = (দশমিক) pa.dm[bia * 2 + 0];
        ধ্রুবক দশমিক dmin_a = (দশমিক) pa.dm[bia * 2 + 1];
        দশমিক dall_b = 0.0f;
        দশমিক dmin_b = 0.0f;
        যদি (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 32>(pb.qh + bib * (QK_K / 8));
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * K_SCALE_আকার);
            dall_b = (দশমিক) pb.dm[bib * 2 + 0];
            dmin_b = (দশমিক) pb.dm[bib * 2 + 1];
        }

        simd<দশমিক, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb += 2) {
            ধ্রুবক পূর্ণসংখ্যা q_offset = sb * 16;
            simd<দশমিক, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<দশমিক, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            ধ্রুবক দশমিক scale_a_lo = scale_f_a[sb];
            ধ্রুবক দশমিক scale_a_hi = scale_f_a[sb + 1];
            ধ্রুবক দশমিক min_a_lo   = min_f_a[sb];
            ধ্রুবক দশমিক min_a_hi   = min_f_a[sb + 1];
            ধ্রুবক দশমিক scale_b_lo = scale_f_b[sb];
            ধ্রুবক দশমিক scale_b_hi = scale_f_b[sb + 1];
            ধ্রুবক দশমিক min_b_lo   = min_f_b[sb];
            ধ্রুবক দশমিক min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo_u8 = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi_u8 = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo_u8 = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi_u8 = qs_hi_b.select<32, 1>(q_offset);
            simd<uint16_t, 32> qa_lo = convert<uint16_t>(qa_lo_u8);
            simd<uint16_t, 32> qa_hi = convert<uint16_t>(qa_hi_u8);
            simd<uint16_t, 32> qb_lo = convert<uint16_t>(qb_lo_u8);
            simd<uint16_t, 32> qb_hi = convert<uint16_t>(qb_hi_u8);

            // add the 5th bit: chunk sb uses qh bit sb, chunk sb+1 uses qh bit sb+1;
            // qh always indexes the same 32 bytes regardless of chunk
            qa_lo += extract_bit_to_pos4(qh_a, sb);
            qa_hi += extract_bit_to_pos4(qh_a, sb + 1);
            qb_lo += extract_bit_to_pos4(qh_b, sb);
            qb_hi += extract_bit_to_pos4(qh_b, sb + 1);

            simd<দশমিক, 32> deq_a_lo = convert<দশমিক>(qa_lo) * scale_a_lo + min_a_lo;
            simd<দশমিক, 32> deq_a_hi = convert<দশমিক>(qa_hi) * scale_a_hi + min_a_hi;
            simd<দশমিক, 32> deq_b_lo = convert<দশমিক>(qb_lo) * scale_b_lo + min_b_lo;
            simd<দশমিক, 32> deq_b_hi = convert<দশমিক>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q6_K, SOA reorder layout:
//   [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales(int8): nb*(QK_K/16)] [d: nb*half]
// ---------------------------------------------------------------------------
template <> কাঠামো esimd_reorder_q_traits<GGML_TYPE_Q6_K> {
    কাঠামো ptrs {
        ধ্রুবক uint8_t *    ql;
        ধ্রুবক uint8_t *    qh;
        ধ্রুবক int8_t *     scales;
        ধ্রুবক sycl::half * d;
    };

    স্থির ESIMD_INLINE ptrs make_ptrs(ধ্রুবক শূন্য * vx, size_t nb) {
        ধ্রুবক uint8_t *    ql     = (ধ্রুবক uint8_t *) vx;
        ধ্রুবক uint8_t *    qh     = ql + nb * (QK_K / 2);
        ধ্রুবক int8_t *     scales = (ধ্রুবক int8_t *) (qh + nb * (QK_K / 4));
        ধ্রুবক sycl::half * d      = (ধ্রুবক sycl::half *) (scales + nb * (QK_K / 16));
        ফেরত { ql, qh, scales, d };
    }

    স্থির ESIMD_INLINE শূন্য mac_pair(
            ধ্রুবক ptrs & pa, size_t bia,
            ধ্রুবক ptrs & pb, size_t bib, যৌক্তিক has_b,
            sycl::ext::intel::esimd::simd<দশমিক, 256> & y_vec,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_a,
            sycl::ext::intel::esimd::simd<দশমিক, 32> & acc_b) {
        using নামাকাশ sycl::ext::intel::esimd;

        simd<uint8_t, 128> ql_a     = block_load<uint8_t, 128>(pa.ql + bia * (QK_K / 2));
        simd<uint8_t, 128> ql_b     = 0;
        simd<uint8_t, 64>  qh_a     = block_load<uint8_t, 64>(pa.qh + bia * (QK_K / 4));
        simd<uint8_t, 64>  qh_b     = 0;
        simd<int8_t, 16>   scales_a = block_load<int8_t, 16>(pa.scales + bia * (QK_K / 16));
        simd<int8_t, 16>   scales_b = 0;

        ধ্রুবক দশমিক d_a = (দশমিক) pa.d[bia];
        দশমিক d_b = 0.0f;
        যদি (has_b) {
            ql_b     = block_load<uint8_t, 128>(pb.ql + bib * (QK_K / 2));
            qh_b     = block_load<uint8_t, 64>(pb.qh + bib * (QK_K / 4));
            scales_b = block_load<int8_t, 16>(pb.scales + bib * (QK_K / 16));
            d_b = (দশমিক) pb.d[bib];
        }

        simd<দশমিক, 16> sc_a = convert<দশমিক>(scales_a);
        simd<দশমিক, 16> sc_b = convert<দশমিক>(scales_b);

#pragma unroll
        জন্য (পূর্ণসংখ্যা im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo_a   = ql_a.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_a   = ql_a.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_a = qh_a.select<32, 1>(32 * im);
            simd<uint8_t, 32> ql_lo_b   = ql_b.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_b   = ql_b.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_b = qh_b.select<32, 1>(32 * im);

            // reconstruct each 32-wide 6-bit group (matches dequantize_row_q6_K)
#pragma unroll
            জন্য (পূর্ণসংখ্যা g = 0; g < 4; ++g) {
                simd<দশমিক, 32> y_g = y_vec.select<32, 1>(32 * (4 * im + g));

                ধ্রুবক দশমিক scale_a_lo = sc_a[8 * im + 2 * g + 0] * d_a;
                ধ্রুবক দশমিক scale_a_hi = sc_a[8 * im + 2 * g + 1] * d_a;
                ধ্রুবক দশমিক scale_b_lo = sc_b[8 * im + 2 * g + 0] * d_b;
                ধ্রুবক দশমিক scale_b_hi = sc_b[8 * im + 2 * g + 1] * d_b;

                simd<দশমিক, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
                simd<দশমিক, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

                simd<uint8_t, 32> qa;
                simd<uint8_t, 32> qb;
                নির্বাচন (g) {
                    ক্ষেত্র 0:
                        qa = (ql_lo_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        qb = (ql_lo_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        বিরতি;
                    ক্ষেত্র 1:
                        qa = (ql_hi_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        qb = (ql_hi_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        বিরতি;
                    ক্ষেত্র 2:
                        qa = (ql_lo_a >> simd<uint8_t, 32>(4)) | (qh_bits_a & simd<uint8_t, 32>(0x30));
                        qb = (ql_lo_b >> simd<uint8_t, 32>(4)) | (qh_bits_b & simd<uint8_t, 32>(0x30));
                        বিরতি;
                    পূর্বনির্ধারিত:
                        qa = (ql_hi_a >> simd<uint8_t, 32>(4)) | ((qh_bits_a & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        qb = (ql_hi_b >> simd<uint8_t, 32>(4)) | ((qh_bits_b & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        বিরতি;
                }

                simd<দশমিক, 32> deq_a = (convert<দশমিক>(qa) - 32.0f) * scale_vec_a;
                simd<দশমিক, 32> deq_b = (convert<দশমিক>(qb) - 32.0f) * scale_vec_b;

                acc_a += y_g * deq_a;
                acc_b += y_g * deq_b;
            }
        }
    }
};

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_ESIMD_HPP
