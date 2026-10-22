#ಅನ್ತರ್ಭಾವ "topk-radix.hpp"

#ಅನ್ತರ್ಭಾವ "common.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>

// Large-k top-k by radix select on an order-preserving unsigned key.
//
// The k-th largest key of a row is found by four most-significant-first passes over its
// 8-bit digits: histogram the digit over the candidate set, walk the buckets from the
// top, and recurse into the bucket where the running count reaches what is still
// needed. Everything strictly above that bucket is in the top-k. A final pass emits
// every column whose key beats the pivot, then exactly as many pivot-equal columns as
// are still missing, so duplicate keys yield exactly k distinct indices.
//
// SLM holds only the histogram, so unlike the scan-merge kernels the cost does not grow
// with k. One work-group owns a row and runs every pass, so a top-k is one launch and
// needs no pool scratch. The row is re-read once per pass rather than compacted, which
// keeps the candidate set implicit: (key & mask) == prefix.
//
// The output is the set of winning indices in no particular order, which is what the
// reference op provides (it swaps its first two outputs to say so) and what
// test-backend-ops compares.

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಬಿಟ್ಟುಗಳು       = 8;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_BUCKETS    = 1 << SYCL_ತುದಿ_K_RADIX_ಬಿಟ್ಟುಗಳು;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_HIST_COPIES = 8;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ   = SYCL_ತುದಿ_K_RADIX_BUCKETS * SYCL_ತುದಿ_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_SLM_ಪದಗಳು   = SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t ತುದಿ_k_radix_ಕೀಲಿ(ದಶಮಾಂಶ f) {
    uint32_t u = sycl::ಬಿಟ್_cast<uint32_t>(f);
    ಆದರೆ (u == 0x80000000u) {
        u = 0u;
    }
    ಹಿಂತಿರುಗಿ (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_radix_select_f32(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
    int32_t *       dst_idx,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k,
    uint32_t *      slm,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1
) {
    ಬಳಸಿ local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::ಸ್ಮರಣೆ_scope::work_ಗುಂಪು,
                                          sycl::access::ವಿಳಾಸ_space::local_space>;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid        = ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಅಂಶ_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ;
    uint32_t * s_ಪಾತ್ರೆ = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 1;
    uint32_t * s_need   = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 2;
    uint32_t * s_cnt_gt = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 3;
    uint32_t * s_cnt_eq = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 4;

    ಆದರೆ (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ copy = tid & (SYCL_ತುದಿ_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t ಮುಖವಾಡ   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸರಿಸು = 32 - SYCL_ತುದಿ_K_RADIX_ಬಿಟ್ಟುಗಳು; ಸರಿಸು >= 0; ಸರಿಸು -= SYCL_ತುದಿ_K_RADIX_ಬಿಟ್ಟುಗಳು) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ; i += ಖಂಡ_ಗಾತ್ರ_3) {
            hist[i] = 0;
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
            ಸ್ಥಿರಾಂಕ uint32_t ಕೀಲಿ = ತುದಿ_k_radix_ಕೀಲಿ(src[ಲಂಬಸಾಲು]);
            ಆದರೆ ((ಕೀಲಿ & ಮುಖವಾಡ) == prefix) {
                ಸ್ಥಿರಾಂಕ uint32_t bucket = (ಕೀಲಿ >> ಸರಿಸು) & (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_ತುದಿ_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        ಆದರೆ (tid < SYCL_ತುದಿ_K_RADIX_BUCKETS) {
            ಸ್ಥಿರಾಂಕ uint32_t * h = hist + (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1 - tid) * SYCL_ತುದಿ_K_RADIX_HIST_COPIES;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < SYCL_ತುದಿ_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        ಸ್ಥಿರಾಂಕ uint32_t incl = sycl::inclusive_ಪರಿಶೀಲನೆ_over_ಗುಂಪು(ಅಂಶ_ct1.get_ಗುಂಪು(), cnt, sycl::plus<uint32_t>());

        ಆದರೆ (tid < SYCL_ತುದಿ_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1 - tid);
            *s_ಪಾತ್ರೆ = cnt;
            *s_need   = need - (incl - cnt);
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        ಸ್ಥಿರಾಂಕ uint32_t digit      = *s_digit;
        ಸ್ಥಿರಾಂಕ uint32_t ಪಾತ್ರೆ_cnt = *s_ಪಾತ್ರೆ;
        need   = *s_need;
        prefix |= digit << ಸರಿಸು;
        ಮುಖವಾಡ   |= (uint32_t) (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1) << ಸರಿಸು;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        ಆದರೆ (ಪಾತ್ರೆ_cnt == need) {
            ನಿಲ್ಲಿಸು;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    ಸ್ಥಿರಾಂಕ uint32_t ತಳ_eq = (uint32_t) k - need;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ uint32_t kp = ತುದಿ_k_radix_ಕೀಲಿ(src[ಲಂಬಸಾಲು]) & ಮುಖವಾಡ;
        ಆದರೆ (kp > prefix) {
            ಸ್ಥಿರಾಂಕ uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = ಲಂಬಸಾಲು;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kp == prefix) {
            ಸ್ಥಿರಾಂಕ uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            ಆದರೆ (pos < need) {
                dst_idx[ತಳ_eq + pos] = ಲಂಬಸಾಲು;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
    int32_t * dst_indices,
    ಸ್ಥಿರಾಂಕ int64_t ncols,
    ಸ್ಥಿರಾಂಕ int64_t nrows,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
    dpct::ಸರತಿ_ptr main_ಹರಿವು
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ggml_sycl_info().max_work_ಗುಂಪು_sizes[ctx.device];
    GGML_ASSERT(ಖಂಡ_ಗಾತ್ರ_3 >= SYCL_ತುದಿ_K_RADIX_BUCKETS);

    ಸ್ಥಿರಾಂಕ sycl::range<1> ಖಂಡ_dims(ಖಂಡ_ಗಾತ್ರ_3);
    ಸ್ಥಿರಾಂಕ sycl::range<1> grid_dims(nrows);

    main_ಹರಿವು->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_ತುದಿ_K_RADIX_SLM_ಪದಗಳು), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(0);

                ತುದಿ_k_radix_select_f32(
                    src + (int64_t) ಅಡ್ಡಸಾಲು * ncols, dst_indices + (int64_t) ಅಡ್ಡಸಾಲು * k,
                    (ಪೂರ್ಣಾಂಕ) ncols, k,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ಅಂಶ_ct1);
            });
    });
}

// One work-group owns a whole row above, which leaves the device idle whenever a graph
// has fewer rows than it has cores -- the common case at batch size 1, where the
// sparse-attention indexer and the backend sampler both top-k a single row. The kernels
// below spread one row over several groups instead.
//
// A digit pass now needs the whole row's histogram before any group can pick the pivot,
// so the per-pass state moves to global memory and the passes become separate launches:
// a work-group barrier no longer spans the row. Each group still accumulates into SLM
// and contributes 256 global atomics at the end, so global traffic is per-group, not
// per-element. The last group to finish a pass (the one whose fetch_add returns G - 1)
// does the scan for the row and clears the histogram for the next pass, which keeps the
// launch count at one per digit rather than two.
//
// Running all four digits unconditionally costs nothing in correctness: once a bucket
// holds exactly the elements still needed, later digits only extend the prefix, and the
// count of columns above that longer prefix grows by exactly as much as `need` shrinks.
// The emit below therefore stays exact whatever pass the answer settled on.

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಗಿದ   = SYCL_ತುದಿ_K_RADIX_BUCKETS + 0;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪೂರ್ವಪ್ರತ್ಯಯ = SYCL_ತುದಿ_K_RADIX_BUCKETS + 1;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಖವಾಡ   = SYCL_ತುದಿ_K_RADIX_BUCKETS + 2;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_NEED   = SYCL_ತುದಿ_K_RADIX_BUCKETS + 3;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_CNT_GT = SYCL_ತುದಿ_K_RADIX_BUCKETS + 4;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_CNT_EQ = SYCL_ತುದಿ_K_RADIX_BUCKETS + 5;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪದಗಳು  = SYCL_ತುದಿ_K_RADIX_BUCKETS + 6;

// How wide the split goes is a property of the device, not of the model: enough groups to
// cover the cores, and no more. Past that the extra groups add histogram traffic without
// adding bandwidth (measured on this device: 20 and 40 groups tie, 60 and 160 lose).
//
// nsm is max_compute_units / 16, i.e. it counts an Xe core as 16 EUs. That is a core's
// width on Xe-HPG, but an Xe2 core is 8 XVEs wide, so on Battlemage the field reads half
// the cores actually present (10 for a 20-core B60). The measured curve is flat from one
// group per core to two and only falls off at three, so a factor of two covers the device
// on Xe2 and lands in the flat region on Xe-HPG. It is the one number here that a correct
// core count would remove; it was tuned on Xe2 and has not been measured on Xe-HPG.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_GROUPS_ಪ್ರತಿ_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_MIN_ವಿಭಜಿಸು_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_RADIX_MIN_PART_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ  = 4096;

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ತುದಿ_k_radix_ವಿಭಜಿಸು_groups(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device, ಸ್ಥಿರಾಂಕ int64_t ncols, ಸ್ಥಿರಾಂಕ int64_t nrows) {
    ಸ್ಥಿರಾಂಕ int64_t ಗುರಿ = (int64_t) SYCL_ತುದಿ_K_RADIX_GROUPS_ಪ್ರತಿ_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    ಆದರೆ (ncols < SYCL_ತುದಿ_K_RADIX_MIN_ವಿಭಜಿಸು_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ || nrows >= ಗುರಿ) {
        ಹಿಂತಿರುಗಿ 1;
    }

    ಸ್ಥಿರಾಂಕ int64_t by_ಅಡ್ಡಸಾಲುಗಳು = ಗುರಿ / nrows;   // floor: never overshoot a row that is nearly covered
    ಸ್ಥಿರಾಂಕ int64_t by_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ncols / SYCL_ತುದಿ_K_RADIX_MIN_PART_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ) std::max<int64_t>(1, std::min(by_ಅಡ್ಡಸಾಲುಗಳು, by_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ));
}

ಬಳಸಿ ತುದಿ_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::ಸ್ಮರಣೆ_scope::device,
                                             sycl::access::ವಿಳಾಸ_space::global_space>;

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_radix_ವಿಭಜಿಸು_pass_f32(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
    uint32_t *      state,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ಸರಿಸು,
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      ಮೊದಲ,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       part,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nparts,
    uint32_t *      slm,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1
) {
    ಬಳಸಿ local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::ಸ್ಮರಣೆ_scope::work_ಗುಂಪು,
                                          sycl::access::ವಿಳಾಸ_space::local_space>;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid        = ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಅಂಶ_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_ಕೊನೆಯ = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ;
    uint32_t * s_ಅಡ್ಡಸಾಲು  = slm + SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    ಆದರೆ (tid == 0) {
        s_ಅಡ್ಡಸಾಲು[0] = ಮೊದಲ ? 0u : state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪೂರ್ವಪ್ರತ್ಯಯ];
        s_ಅಡ್ಡಸಾಲು[1] = ಮೊದಲ ? 0u : state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಖವಾಡ];
        s_ಅಡ್ಡಸಾಲು[2] = ಮೊದಲ ? (uint32_t) k : state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_NEED];
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ; i += ಖಂಡ_ಗಾತ್ರ_3) {
        hist[i] = 0;
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಸ್ಥಿರಾಂಕ uint32_t prefix = s_ಅಡ್ಡಸಾಲು[0];
    ಸ್ಥಿರಾಂಕ uint32_t ಮುಖವಾಡ   = s_ಅಡ್ಡಸಾಲು[1];
    ಸ್ಥಿರಾಂಕ uint32_t need   = s_ಅಡ್ಡಸಾಲು[2];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ copy  = tid & (SYCL_ತುದಿ_K_RADIX_HIST_COPIES - 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk = (ncols + nparts - 1) / nparts;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col0  = part * chunk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col1  = std::min(ncols, col0 + chunk);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = col0 + tid; ಲಂಬಸಾಲು < col1; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ uint32_t ಕೀಲಿ = ತುದಿ_k_radix_ಕೀಲಿ(src[ಲಂಬಸಾಲು]);
        ಆದರೆ ((ಕೀಲಿ & ಮುಖವಾಡ) == prefix) {
            ಸ್ಥಿರಾಂಕ uint32_t bucket = (ಕೀಲಿ >> ಸರಿಸು) & (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_ತುದಿ_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = tid; b < SYCL_ತುದಿ_K_RADIX_BUCKETS; b += ಖಂಡ_ಗಾತ್ರ_3) {
        uint32_t ಮೊತ್ತ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < SYCL_ತುದಿ_K_RADIX_HIST_COPIES; c++) {
            ಮೊತ್ತ += hist[b * SYCL_ತುದಿ_K_RADIX_HIST_COPIES + c];
        }
        ಆದರೆ (ಮೊತ್ತ) {
            ತುದಿ_k_radix_gatomic(state[b]).fetch_add(ಮೊತ್ತ);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::global_and_local);
    ಆದರೆ (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::ಸ್ಮರಣೆ_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::ಸ್ಮರಣೆ_scope::device,
                         sycl::access::ವಿಳಾಸ_space::global_space> ಮುಗಿದ(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಗಿದ]);
        *s_ಕೊನೆಯ = (ಮುಗಿದ.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಆದರೆ (*s_ಕೊನೆಯ == 0u) {
        ಹಿಂತಿರುಗಿ;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::ಸ್ಮರಣೆ_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    ಆದರೆ (tid < SYCL_ತುದಿ_K_RADIX_BUCKETS) {
        cnt = ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    ಸ್ಥಿರಾಂಕ uint32_t incl = sycl::inclusive_ಪರಿಶೀಲನೆ_over_ಗುಂಪು(ಅಂಶ_ct1.get_ಗುಂಪು(), cnt, sycl::plus<uint32_t>());

    ಆದರೆ (tid < SYCL_ತುದಿ_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        ಸ್ಥಿರಾಂಕ uint32_t digit = (uint32_t) (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1 - tid);
        ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪೂರ್ವಪ್ರತ್ಯಯ]).store(prefix | (digit << ಸರಿಸು));
        ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಖವಾಡ]).store(
            ಮುಖವಾಡ | ((uint32_t) (SYCL_ತುದಿ_K_RADIX_BUCKETS - 1) << ಸರಿಸು));
        ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_NEED]).store(need - (incl - cnt));
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = tid; b < SYCL_ತುದಿ_K_RADIX_BUCKETS; b += ಖಂಡ_ಗಾತ್ರ_3) {
        ತುದಿ_k_radix_gatomic(state[b]).store(0u);
    }
    ಆದರೆ (tid == 0) {
        ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಗಿದ]).store(0u);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_radix_ವಿಭಜಿಸು_emit_f32(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
    int32_t *       dst_idx,
    uint32_t *      state,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       part,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nparts,
    uint32_t *      slm,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1
) {
    ಬಳಸಿ local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::ಸ್ಮರಣೆ_scope::work_ಗುಂಪು,
                                          sycl::access::ವಿಳಾಸ_space::local_space>;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid        = ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಅಂಶ_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_ತಳ_gt = slm + 2;
    uint32_t * s_ತಳ_eq = slm + 3;

    uint32_t * s_ಅಡ್ಡಸಾಲು = slm + 4;   // prefix, mask, need

    ಆದರೆ (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_ಅಡ್ಡಸಾಲು[0] = state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪೂರ್ವಪ್ರತ್ಯಯ];
        s_ಅಡ್ಡಸಾಲು[1] = state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಮುಖವಾಡ];
        s_ಅಡ್ಡಸಾಲು[2] = state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_NEED];
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಸ್ಥಿರಾಂಕ uint32_t prefix = s_ಅಡ್ಡಸಾಲು[0];
    ಸ್ಥಿರಾಂಕ uint32_t ಮುಖವಾಡ   = s_ಅಡ್ಡಸಾಲು[1];
    ಸ್ಥಿರಾಂಕ uint32_t need   = s_ಅಡ್ಡಸಾಲು[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    ಸ್ಥಿರಾಂಕ uint32_t ತಳ_eq = (uint32_t) k - need;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk = (ncols + nparts - 1) / nparts;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col0  = part * chunk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = col0 + tid; ಲಂಬಸಾಲು < col1; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ uint32_t kp = ತುದಿ_k_radix_ಕೀಲಿ(src[ಲಂಬಸಾಲು]) & ಮುಖವಾಡ;
        ಆದರೆ (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಆದರೆ (tid == 0) {
        ಸ್ಥಿರಾಂಕ uint32_t n_gt = *s_gt;
        ಸ್ಥಿರಾಂಕ uint32_t n_eq = *s_eq;
        *s_ತಳ_gt = n_gt ? ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_ತಳ_eq = n_eq ? ತುದಿ_k_radix_gatomic(state[SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಸ್ಥಿರಾಂಕ uint32_t ತಳ_gt_g = *s_ತಳ_gt;
    ಸ್ಥಿರಾಂಕ uint32_t ತಳ_eq_g = *s_ತಳ_eq;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = col0 + tid; ಲಂಬಸಾಲು < col1; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ uint32_t kp = ತುದಿ_k_radix_ಕೀಲಿ(src[ಲಂಬಸಾಲು]) & ಮುಖವಾಡ;
        ಆದರೆ (kp > prefix) {
            dst_idx[ತಳ_gt_g + local_atomic(*s_gt).fetch_add(1u)] = ಲಂಬಸಾಲು;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kp == prefix) {
            ಸ್ಥಿರಾಂಕ uint32_t pos = ತಳ_eq_g + local_atomic(*s_eq).fetch_add(1u);
            ಆದರೆ (pos < need) {
                dst_idx[ತಳ_eq + pos] = ಲಂಬಸಾಲು;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_radix_ವಿಭಜಿಸು_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
    int32_t * dst_indices,
    ಸ್ಥಿರಾಂಕ int64_t ncols,
    ಸ್ಥಿರಾಂಕ int64_t nrows,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nparts,
    dpct::ಸರತಿ_ptr main_ಹರಿವು
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ggml_sycl_info().max_work_ಗುಂಪು_sizes[ctx.device];
    GGML_ASSERT(ಖಂಡ_ಗಾತ್ರ_3 >= SYCL_ತುದಿ_K_RADIX_BUCKETS);

    ಸ್ಥಿರಾಂಕ size_t state_ಪದಗಳು = (size_t) nrows * SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪದಗಳು;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_ಪದಗಳು);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_ಹರಿವು->memset(state, 0, state_ಪದಗಳು * ಗಾತ್ರ(uint32_t))));

    ಸ್ಥಿರಾಂಕ sycl::range<1> ಖಂಡ_dims(ಖಂಡ_ಗಾತ್ರ_3);
    ಸ್ಥಿರಾಂಕ sycl::range<1> grid_dims(nrows * nparts);

    ತಾರ್ಕಿಕ ಮೊದಲ = ನಿಜ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸರಿಸು = 32 - SYCL_ತುದಿ_K_RADIX_ಬಿಟ್ಟುಗಳು; ಸರಿಸು >= 0; ಸರಿಸು -= SYCL_ತುದಿ_K_RADIX_ಬಿಟ್ಟುಗಳು) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಮೊದಲ = ಮೊದಲ;
        ಮೊದಲ = ಸುಳ್ಳು;
        main_ಹರಿವು->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_ತುದಿ_K_RADIX_HIST_ಗಾತ್ರ + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g    = ಅಂಶ_ct1.get_ಗುಂಪು(0);
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು  = g / nparts;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ part = g % nparts;

                    ತುದಿ_k_radix_ವಿಭಜಿಸು_pass_f32(
                        src + (int64_t) ಅಡ್ಡಸಾಲು * ncols,
                        state + (int64_t) ಅಡ್ಡಸಾಲು * SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪದಗಳು,
                        (ಪೂರ್ಣಾಂಕ) ncols, k, ಸರಿಸು, is_ಮೊದಲ, part, nparts,
                        slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                        ಅಂಶ_ct1);
                });
        });
    }

    main_ಹರಿವು->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(8), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g    = ಅಂಶ_ct1.get_ಗುಂಪು(0);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು  = g / nparts;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ part = g % nparts;

                ತುದಿ_k_radix_ವಿಭಜಿಸು_emit_f32(
                    src + (int64_t) ಅಡ್ಡಸಾಲು * ncols,
                    dst_indices + (int64_t) ಅಡ್ಡಸಾಲು * k,
                    state + (int64_t) ಅಡ್ಡಸಾಲು * SYCL_ತುದಿ_K_RADIX_ಅಡ್ಡಸಾಲು_ಪದಗಳು,
                    (ಪೂರ್ಣಾಂಕ) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ಅಂಶ_ct1);
            });
    });
}

ಶೂನ್ಯ ggml_sycl_ತುದಿ_k_radix(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
    int32_t *       dst_indices,
    ಸ್ಥಿರಾಂಕ int64_t   ncols,
    ಸ್ಥಿರಾಂಕ int64_t   nrows,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k,
    dpct::ಸರತಿ_ptr main_ಹರಿವು
) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nparts = ತುದಿ_k_radix_ವಿಭಜಿಸು_groups(ctx.device, ncols, nrows);
    ಆದರೆ (nparts > 1) {
        ತುದಿ_k_radix_ವಿಭಜಿಸು_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ತುದಿ_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_ಹರಿವು);
    }
}
