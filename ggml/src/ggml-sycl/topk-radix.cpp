#ضمن "topk-radix.hpp"

#ضمن "common.hpp"

#ضمن <algorithm>

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

ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_BITS       = 8;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_BUCKETS    = 1 << SYCL_چوٹی_K_RADIX_BITS;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_HIST_COPIES = 8;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_HIST_SIZE   = SYCL_چوٹی_K_RADIX_BUCKETS * SYCL_چوٹی_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_SLM_الفاظ   = SYCL_چوٹی_K_RADIX_HIST_SIZE + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
ساكن مضمن uint32_t چوٹی_k_radix_کلید(عائم f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    إذا (u == 0x80000000u) {
        u = 0u;
    }
    أرجع (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

ساكن فراغ چوٹی_k_radix_select_f32(
    ثابت عائم *   src,
    int32_t *       dst_idx,
    ثابت صحيح       ncols,
    ثابت صحيح       k,
    uint32_t *      slm,
    ثابت sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    ثابت صحيح tid        = item_ct1.get_local_id(0);
    ثابت صحيح block_size = item_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE;
    uint32_t * s_bucket = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE + 1;
    uint32_t * s_need   = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE + 2;
    uint32_t * s_cnt_gt = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE + 3;
    uint32_t * s_cnt_eq = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE + 4;

    إذا (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    ثابت صحيح copy = tid & (SYCL_چوٹی_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t mask   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    لكل (صحيح shift = 32 - SYCL_چوٹی_K_RADIX_BITS; shift >= 0; shift -= SYCL_چوٹی_K_RADIX_BITS) {
        لكل (صحيح i = tid; i < SYCL_چوٹی_K_RADIX_HIST_SIZE; i += block_size) {
            hist[i] = 0;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        لكل (صحيح col = tid; col < ncols; col += block_size) {
            ثابت uint32_t کلید = چوٹی_k_radix_کلید(src[col]);
            إذا ((کلید & mask) == prefix) {
                ثابت uint32_t bucket = (کلید >> shift) & (SYCL_چوٹی_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_چوٹی_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS) {
            ثابت uint32_t * h = hist + (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid) * SYCL_چوٹی_K_RADIX_HIST_COPIES;
            لكل (صحيح c = 0; c < SYCL_چوٹی_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        ثابت uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());

        إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid);
            *s_bucket = cnt;
            *s_need   = need - (incl - cnt);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت uint32_t digit      = *s_digit;
        ثابت uint32_t bucket_cnt = *s_bucket;
        need   = *s_need;
        prefix |= digit << shift;
        mask   |= (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1) << shift;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        إذا (bucket_cnt == need) {
            اكسر;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    ثابت uint32_t base_eq = (uint32_t) k - need;

    لكل (صحيح col = tid; col < ncols; col += block_size) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[col]) & mask;
        إذا (kp > prefix) {
            ثابت uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = col;
        } وإلا إذا (kp == prefix) {
            ثابت uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            إذا (pos < need) {
                dst_idx[base_eq + pos] = col;
            }
        }
    }
}

ساكن فراغ چوٹی_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت عائم * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحيح k,
    dpct::queue_ptr main_stream
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    ثابت صحيح block_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
    GGML_ASSERT(block_size >= SYCL_چوٹی_K_RADIX_BUCKETS);

    ثابت sycl::range<1> block_dims(block_size);
    ثابت sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_چوٹی_K_RADIX_SLM_الفاظ), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                ثابت صحيح row = item_ct1.get_group(0);

                چوٹی_k_radix_select_f32(
                    src + (int64_t) row * ncols, dst_indices + (int64_t) row * k,
                    (صحيح) ncols, k,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item_ct1);
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

ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_DONE   = SYCL_چوٹی_K_RADIX_BUCKETS + 0;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_PREFIX = SYCL_چوٹی_K_RADIX_BUCKETS + 1;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_MASK   = SYCL_چوٹی_K_RADIX_BUCKETS + 2;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_NEED   = SYCL_چوٹی_K_RADIX_BUCKETS + 3;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_CNT_GT = SYCL_چوٹی_K_RADIX_BUCKETS + 4;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_CNT_EQ = SYCL_چوٹی_K_RADIX_BUCKETS + 5;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_ROW_الفاظ  = SYCL_چوٹی_K_RADIX_BUCKETS + 6;

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
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_GROUPS_PER_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_MIN_SPLIT_COLS = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_MIN_PART_COLS  = 4096;

ساكن صحيح چوٹی_k_radix_split_groups(ثابت صحيح device, ثابت int64_t ncols, ثابت int64_t nrows) {
    ثابت int64_t target = (int64_t) SYCL_چوٹی_K_RADIX_GROUPS_PER_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    إذا (ncols < SYCL_چوٹی_K_RADIX_MIN_SPLIT_COLS || nrows >= target) {
        أرجع 1;
    }

    ثابت int64_t by_rows = target / nrows;   // floor: never overshoot a row that is nearly covered
    ثابت int64_t by_cols = ncols / SYCL_چوٹی_K_RADIX_MIN_PART_COLS;

    أرجع (صحيح) std::max<int64_t>(1, std::min(by_rows, by_cols));
}

using چوٹی_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

ساكن فراغ چوٹی_k_radix_split_pass_f32(
    ثابت عائم *   src,
    uint32_t *      state,
    ثابت صحيح       ncols,
    ثابت صحيح       k,
    ثابت صحيح       shift,
    ثابت منطقي      first,
    ثابت صحيح       part,
    ثابت صحيح       nparts,
    uint32_t *      slm,
    ثابت sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    ثابت صحيح tid        = item_ct1.get_local_id(0);
    ثابت صحيح block_size = item_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_last = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE;
    uint32_t * s_row  = slm + SYCL_چوٹی_K_RADIX_HIST_SIZE + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    إذا (tid == 0) {
        s_row[0] = first ? 0u : state[SYCL_چوٹی_K_RADIX_ROW_PREFIX];
        s_row[1] = first ? 0u : state[SYCL_چوٹی_K_RADIX_ROW_MASK];
        s_row[2] = first ? (uint32_t) k : state[SYCL_چوٹی_K_RADIX_ROW_NEED];
    }

    لكل (صحيح i = tid; i < SYCL_چوٹی_K_RADIX_HIST_SIZE; i += block_size) {
        hist[i] = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t prefix = s_row[0];
    ثابت uint32_t mask   = s_row[1];
    ثابت uint32_t need   = s_row[2];

    ثابت صحيح copy  = tid & (SYCL_چوٹی_K_RADIX_HIST_COPIES - 1);
    ثابت صحيح chunk = (ncols + nparts - 1) / nparts;
    ثابت صحيح col0  = part * chunk;
    ثابت صحيح col1  = std::min(ncols, col0 + chunk);

    لكل (صحيح col = col0 + tid; col < col1; col += block_size) {
        ثابت uint32_t کلید = چوٹی_k_radix_کلید(src[col]);
        إذا ((کلید & mask) == prefix) {
            ثابت uint32_t bucket = (کلید >> shift) & (SYCL_چوٹی_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_چوٹی_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    لكل (صحيح b = tid; b < SYCL_چوٹی_K_RADIX_BUCKETS; b += block_size) {
        uint32_t sum = 0;
        لكل (صحيح c = 0; c < SYCL_چوٹی_K_RADIX_HIST_COPIES; c++) {
            sum += hist[b * SYCL_چوٹی_K_RADIX_HIST_COPIES + c];
        }
        إذا (sum) {
            چوٹی_k_radix_gatomic(state[b]).fetch_add(sum);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
    إذا (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> done(state[SYCL_چوٹی_K_RADIX_ROW_DONE]);
        *s_last = (done.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    إذا (*s_last == 0u) {
        أرجع;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS) {
        cnt = چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    ثابت uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());

    إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        ثابت uint32_t digit = (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid);
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_ROW_PREFIX]).store(prefix | (digit << shift));
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_ROW_MASK]).store(
            mask | ((uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1) << shift));
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_ROW_NEED]).store(need - (incl - cnt));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    لكل (صحيح b = tid; b < SYCL_چوٹی_K_RADIX_BUCKETS; b += block_size) {
        چوٹی_k_radix_gatomic(state[b]).store(0u);
    }
    إذا (tid == 0) {
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_ROW_DONE]).store(0u);
    }
}

ساكن فراغ چوٹی_k_radix_split_emit_f32(
    ثابت عائم *   src,
    int32_t *       dst_idx,
    uint32_t *      state,
    ثابت صحيح       ncols,
    ثابت صحيح       k,
    ثابت صحيح       part,
    ثابت صحيح       nparts,
    uint32_t *      slm,
    ثابت sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    ثابت صحيح tid        = item_ct1.get_local_id(0);
    ثابت صحيح block_size = item_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_base_gt = slm + 2;
    uint32_t * s_base_eq = slm + 3;

    uint32_t * s_row = slm + 4;   // prefix, mask, need

    إذا (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_row[0] = state[SYCL_چوٹی_K_RADIX_ROW_PREFIX];
        s_row[1] = state[SYCL_چوٹی_K_RADIX_ROW_MASK];
        s_row[2] = state[SYCL_چوٹی_K_RADIX_ROW_NEED];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t prefix = s_row[0];
    ثابت uint32_t mask   = s_row[1];
    ثابت uint32_t need   = s_row[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    ثابت uint32_t base_eq = (uint32_t) k - need;

    ثابت صحيح chunk = (ncols + nparts - 1) / nparts;
    ثابت صحيح col0  = part * chunk;
    ثابت صحيح col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    لكل (صحيح col = col0 + tid; col < col1; col += block_size) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[col]) & mask;
        إذا (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } وإلا إذا (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    إذا (tid == 0) {
        ثابت uint32_t n_gt = *s_gt;
        ثابت uint32_t n_eq = *s_eq;
        *s_base_gt = n_gt ? چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_ROW_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_base_eq = n_eq ? چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_ROW_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t base_gt_g = *s_base_gt;
    ثابت uint32_t base_eq_g = *s_base_eq;

    لكل (صحيح col = col0 + tid; col < col1; col += block_size) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[col]) & mask;
        إذا (kp > prefix) {
            dst_idx[base_gt_g + local_atomic(*s_gt).fetch_add(1u)] = col;
        } وإلا إذا (kp == prefix) {
            ثابت uint32_t pos = base_eq_g + local_atomic(*s_eq).fetch_add(1u);
            إذا (pos < need) {
                dst_idx[base_eq + pos] = col;
            }
        }
    }
}

ساكن فراغ چوٹی_k_radix_split_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت عائم * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحيح k,
    ثابت صحيح nparts,
    dpct::queue_ptr main_stream
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    ثابت صحيح block_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
    GGML_ASSERT(block_size >= SYCL_چوٹی_K_RADIX_BUCKETS);

    ثابت size_t state_الفاظ = (size_t) nrows * SYCL_چوٹی_K_RADIX_ROW_الفاظ;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_الفاظ);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_stream->memset(state, 0, state_الفاظ * حجم(uint32_t))));

    ثابت sycl::range<1> block_dims(block_size);
    ثابت sycl::range<1> grid_dims(nrows * nparts);

    منطقي first = true;
    لكل (صحيح shift = 32 - SYCL_چوٹی_K_RADIX_BITS; shift >= 0; shift -= SYCL_چوٹی_K_RADIX_BITS) {
        ثابت منطقي is_first = first;
        first = خطأ;
        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_چوٹی_K_RADIX_HIST_SIZE + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    ثابت صحيح g    = item_ct1.get_group(0);
                    ثابت صحيح row  = g / nparts;
                    ثابت صحيح part = g % nparts;

                    چوٹی_k_radix_split_pass_f32(
                        src + (int64_t) row * ncols,
                        state + (int64_t) row * SYCL_چوٹی_K_RADIX_ROW_الفاظ,
                        (صحيح) ncols, k, shift, is_first, part, nparts,
                        slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item_ct1);
                });
        });
    }

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(8), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                ثابت صحيح g    = item_ct1.get_group(0);
                ثابت صحيح row  = g / nparts;
                ثابت صحيح part = g % nparts;

                چوٹی_k_radix_split_emit_f32(
                    src + (int64_t) row * ncols,
                    dst_indices + (int64_t) row * k,
                    state + (int64_t) row * SYCL_چوٹی_K_RADIX_ROW_الفاظ,
                    (صحيح) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item_ct1);
            });
    });
}

فراغ ggml_sycl_چوٹی_k_radix(
    ggml_backend_sycl_context & ctx,
    ثابت عائم *   src,
    int32_t *       dst_indices,
    ثابت int64_t   ncols,
    ثابت int64_t   nrows,
    ثابت صحيح       k,
    dpct::queue_ptr main_stream
) {
    ثابت صحيح nparts = چوٹی_k_radix_split_groups(ctx.device, ncols, nrows);
    إذا (nparts > 1) {
        چوٹی_k_radix_split_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_stream);
    } وإلا {
        چوٹی_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_stream);
    }
}
