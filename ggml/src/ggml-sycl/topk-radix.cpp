#அந்தர்பாவ "topk-radix.hpp"

#அந்தர்பாவ "common.hpp"

#அந்தர்பாவ <algorithm>

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

நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_BITS       = 8;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_BUCKETS    = 1 << SYCL_உச்சி_K_RADIX_BITS;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_HIST_COPIES = 8;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_HIST_SIZE   = SYCL_உச்சி_K_RADIX_BUCKETS * SYCL_உச்சி_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_SLM_சொற்கள்   = SYCL_உச்சி_K_RADIX_HIST_SIZE + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
நிலையான உள்வரி uint32_t உச்சி_k_radix_சாவி(தசம f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    எனில் (u == 0x80000000u) {
        u = 0u;
    }
    திரும்பு (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

நிலையான வெறுமை உச்சி_k_radix_select_f32(
    மாறிலி தசம *   src,
    int32_t *       dst_idx,
    மாறிலி முழு       ncols,
    மாறிலி முழு       k,
    uint32_t *      slm,
    மாறிலி sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    மாறிலி முழு tid        = item_ct1.get_local_id(0);
    மாறிலி முழு block_size = item_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE;
    uint32_t * s_bucket = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE + 1;
    uint32_t * s_need   = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE + 2;
    uint32_t * s_cnt_gt = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE + 3;
    uint32_t * s_cnt_eq = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE + 4;

    எனில் (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    மாறிலி முழு copy = tid & (SYCL_உச்சி_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t mask   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    க்கு (முழு shift = 32 - SYCL_உச்சி_K_RADIX_BITS; shift >= 0; shift -= SYCL_உச்சி_K_RADIX_BITS) {
        க்கு (முழு i = tid; i < SYCL_உச்சி_K_RADIX_HIST_SIZE; i += block_size) {
            hist[i] = 0;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        க்கு (முழு col = tid; col < ncols; col += block_size) {
            மாறிலி uint32_t சாவி = உச்சி_k_radix_சாவி(src[col]);
            எனில் ((சாவி & mask) == prefix) {
                மாறிலி uint32_t bucket = (சாவி >> shift) & (SYCL_உச்சி_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_உச்சி_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        எனில் (tid < SYCL_உச்சி_K_RADIX_BUCKETS) {
            மாறிலி uint32_t * h = hist + (SYCL_உச்சி_K_RADIX_BUCKETS - 1 - tid) * SYCL_உச்சி_K_RADIX_HIST_COPIES;
            க்கு (முழு c = 0; c < SYCL_உச்சி_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        மாறிலி uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());

        எனில் (tid < SYCL_உச்சி_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_உச்சி_K_RADIX_BUCKETS - 1 - tid);
            *s_bucket = cnt;
            *s_need   = need - (incl - cnt);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        மாறிலி uint32_t digit      = *s_digit;
        மாறிலி uint32_t bucket_cnt = *s_bucket;
        need   = *s_need;
        prefix |= digit << shift;
        mask   |= (uint32_t) (SYCL_உச்சி_K_RADIX_BUCKETS - 1) << shift;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        எனில் (bucket_cnt == need) {
            நிறுத்து;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    மாறிலி uint32_t base_eq = (uint32_t) k - need;

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        மாறிலி uint32_t kp = உச்சி_k_radix_சாவி(src[col]) & mask;
        எனில் (kp > prefix) {
            மாறிலி uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = col;
        } இல்லையெனில் எனில் (kp == prefix) {
            மாறிலி uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            எனில் (pos < need) {
                dst_idx[base_eq + pos] = col;
            }
        }
    }
}

நிலையான வெறுமை உச்சி_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    மாறிலி தசம * src,
    int32_t * dst_indices,
    மாறிலி int64_t ncols,
    மாறிலி int64_t nrows,
    மாறிலி முழு k,
    dpct::queue_ptr main_stream
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    மாறிலி முழு block_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
    GGML_ASSERT(block_size >= SYCL_உச்சி_K_RADIX_BUCKETS);

    மாறிலி sycl::range<1> block_dims(block_size);
    மாறிலி sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_உச்சி_K_RADIX_SLM_சொற்கள்), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                மாறிலி முழு row = item_ct1.get_group(0);

                உச்சி_k_radix_select_f32(
                    src + (int64_t) row * ncols, dst_indices + (int64_t) row * k,
                    (முழு) ncols, k,
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

நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_DONE   = SYCL_உச்சி_K_RADIX_BUCKETS + 0;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_PREFIX = SYCL_உச்சி_K_RADIX_BUCKETS + 1;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_MASK   = SYCL_உச்சி_K_RADIX_BUCKETS + 2;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_NEED   = SYCL_உச்சி_K_RADIX_BUCKETS + 3;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_CNT_GT = SYCL_உச்சி_K_RADIX_BUCKETS + 4;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_CNT_EQ = SYCL_உச்சி_K_RADIX_BUCKETS + 5;
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_ROW_சொற்கள்  = SYCL_உச்சி_K_RADIX_BUCKETS + 6;

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
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_GROUPS_PER_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_MIN_SPLIT_COLS = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
நிலையான constexpr முழு SYCL_உச்சி_K_RADIX_MIN_PART_COLS  = 4096;

நிலையான முழு உச்சி_k_radix_split_groups(மாறிலி முழு device, மாறிலி int64_t ncols, மாறிலி int64_t nrows) {
    மாறிலி int64_t target = (int64_t) SYCL_உச்சி_K_RADIX_GROUPS_PER_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    எனில் (ncols < SYCL_உச்சி_K_RADIX_MIN_SPLIT_COLS || nrows >= target) {
        திரும்பு 1;
    }

    மாறிலி int64_t by_rows = target / nrows;   // floor: never overshoot a row that is nearly covered
    மாறிலி int64_t by_cols = ncols / SYCL_உச்சி_K_RADIX_MIN_PART_COLS;

    திரும்பு (முழு) std::max<int64_t>(1, std::min(by_rows, by_cols));
}

using உச்சி_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

நிலையான வெறுமை உச்சி_k_radix_split_pass_f32(
    மாறிலி தசம *   src,
    uint32_t *      state,
    மாறிலி முழு       ncols,
    மாறிலி முழு       k,
    மாறிலி முழு       shift,
    மாறிலி தருக்க      first,
    மாறிலி முழு       part,
    மாறிலி முழு       nparts,
    uint32_t *      slm,
    மாறிலி sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    மாறிலி முழு tid        = item_ct1.get_local_id(0);
    மாறிலி முழு block_size = item_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_last = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE;
    uint32_t * s_row  = slm + SYCL_உச்சி_K_RADIX_HIST_SIZE + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    எனில் (tid == 0) {
        s_row[0] = first ? 0u : state[SYCL_உச்சி_K_RADIX_ROW_PREFIX];
        s_row[1] = first ? 0u : state[SYCL_உச்சி_K_RADIX_ROW_MASK];
        s_row[2] = first ? (uint32_t) k : state[SYCL_உச்சி_K_RADIX_ROW_NEED];
    }

    க்கு (முழு i = tid; i < SYCL_உச்சி_K_RADIX_HIST_SIZE; i += block_size) {
        hist[i] = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    மாறிலி uint32_t prefix = s_row[0];
    மாறிலி uint32_t mask   = s_row[1];
    மாறிலி uint32_t need   = s_row[2];

    மாறிலி முழு copy  = tid & (SYCL_உச்சி_K_RADIX_HIST_COPIES - 1);
    மாறிலி முழு chunk = (ncols + nparts - 1) / nparts;
    மாறிலி முழு col0  = part * chunk;
    மாறிலி முழு col1  = std::min(ncols, col0 + chunk);

    க்கு (முழு col = col0 + tid; col < col1; col += block_size) {
        மாறிலி uint32_t சாவி = உச்சி_k_radix_சாவி(src[col]);
        எனில் ((சாவி & mask) == prefix) {
            மாறிலி uint32_t bucket = (சாவி >> shift) & (SYCL_உச்சி_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_உச்சி_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    க்கு (முழு b = tid; b < SYCL_உச்சி_K_RADIX_BUCKETS; b += block_size) {
        uint32_t sum = 0;
        க்கு (முழு c = 0; c < SYCL_உச்சி_K_RADIX_HIST_COPIES; c++) {
            sum += hist[b * SYCL_உச்சி_K_RADIX_HIST_COPIES + c];
        }
        எனில் (sum) {
            உச்சி_k_radix_gatomic(state[b]).fetch_add(sum);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
    எனில் (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> done(state[SYCL_உச்சி_K_RADIX_ROW_DONE]);
        *s_last = (done.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    எனில் (*s_last == 0u) {
        திரும்பு;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    எனில் (tid < SYCL_உச்சி_K_RADIX_BUCKETS) {
        cnt = உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    மாறிலி uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());

    எனில் (tid < SYCL_உச்சி_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        மாறிலி uint32_t digit = (uint32_t) (SYCL_உச்சி_K_RADIX_BUCKETS - 1 - tid);
        உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_ROW_PREFIX]).store(prefix | (digit << shift));
        உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_ROW_MASK]).store(
            mask | ((uint32_t) (SYCL_உச்சி_K_RADIX_BUCKETS - 1) << shift));
        உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_ROW_NEED]).store(need - (incl - cnt));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    க்கு (முழு b = tid; b < SYCL_உச்சி_K_RADIX_BUCKETS; b += block_size) {
        உச்சி_k_radix_gatomic(state[b]).store(0u);
    }
    எனில் (tid == 0) {
        உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_ROW_DONE]).store(0u);
    }
}

நிலையான வெறுமை உச்சி_k_radix_split_emit_f32(
    மாறிலி தசம *   src,
    int32_t *       dst_idx,
    uint32_t *      state,
    மாறிலி முழு       ncols,
    மாறிலி முழு       k,
    மாறிலி முழு       part,
    மாறிலி முழு       nparts,
    uint32_t *      slm,
    மாறிலி sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    மாறிலி முழு tid        = item_ct1.get_local_id(0);
    மாறிலி முழு block_size = item_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_base_gt = slm + 2;
    uint32_t * s_base_eq = slm + 3;

    uint32_t * s_row = slm + 4;   // prefix, mask, need

    எனில் (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_row[0] = state[SYCL_உச்சி_K_RADIX_ROW_PREFIX];
        s_row[1] = state[SYCL_உச்சி_K_RADIX_ROW_MASK];
        s_row[2] = state[SYCL_உச்சி_K_RADIX_ROW_NEED];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    மாறிலி uint32_t prefix = s_row[0];
    மாறிலி uint32_t mask   = s_row[1];
    மாறிலி uint32_t need   = s_row[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    மாறிலி uint32_t base_eq = (uint32_t) k - need;

    மாறிலி முழு chunk = (ncols + nparts - 1) / nparts;
    மாறிலி முழு col0  = part * chunk;
    மாறிலி முழு col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    க்கு (முழு col = col0 + tid; col < col1; col += block_size) {
        மாறிலி uint32_t kp = உச்சி_k_radix_சாவி(src[col]) & mask;
        எனில் (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } இல்லையெனில் எனில் (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    எனில் (tid == 0) {
        மாறிலி uint32_t n_gt = *s_gt;
        மாறிலி uint32_t n_eq = *s_eq;
        *s_base_gt = n_gt ? உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_ROW_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_base_eq = n_eq ? உச்சி_k_radix_gatomic(state[SYCL_உச்சி_K_RADIX_ROW_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    மாறிலி uint32_t base_gt_g = *s_base_gt;
    மாறிலி uint32_t base_eq_g = *s_base_eq;

    க்கு (முழு col = col0 + tid; col < col1; col += block_size) {
        மாறிலி uint32_t kp = உச்சி_k_radix_சாவி(src[col]) & mask;
        எனில் (kp > prefix) {
            dst_idx[base_gt_g + local_atomic(*s_gt).fetch_add(1u)] = col;
        } இல்லையெனில் எனில் (kp == prefix) {
            மாறிலி uint32_t pos = base_eq_g + local_atomic(*s_eq).fetch_add(1u);
            எனில் (pos < need) {
                dst_idx[base_eq + pos] = col;
            }
        }
    }
}

நிலையான வெறுமை உச்சி_k_radix_split_f32_sycl(
    ggml_backend_sycl_context & ctx,
    மாறிலி தசம * src,
    int32_t * dst_indices,
    மாறிலி int64_t ncols,
    மாறிலி int64_t nrows,
    மாறிலி முழு k,
    மாறிலி முழு nparts,
    dpct::queue_ptr main_stream
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    மாறிலி முழு block_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
    GGML_ASSERT(block_size >= SYCL_உச்சி_K_RADIX_BUCKETS);

    மாறிலி size_t state_சொற்கள் = (size_t) nrows * SYCL_உச்சி_K_RADIX_ROW_சொற்கள்;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_சொற்கள்);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_stream->memset(state, 0, state_சொற்கள் * அளவு(uint32_t))));

    மாறிலி sycl::range<1> block_dims(block_size);
    மாறிலி sycl::range<1> grid_dims(nrows * nparts);

    தருக்க first = உண்மை;
    க்கு (முழு shift = 32 - SYCL_உச்சி_K_RADIX_BITS; shift >= 0; shift -= SYCL_உச்சி_K_RADIX_BITS) {
        மாறிலி தருக்க is_first = first;
        first = பொய்;
        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_உச்சி_K_RADIX_HIST_SIZE + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    மாறிலி முழு g    = item_ct1.get_group(0);
                    மாறிலி முழு row  = g / nparts;
                    மாறிலி முழு part = g % nparts;

                    உச்சி_k_radix_split_pass_f32(
                        src + (int64_t) row * ncols,
                        state + (int64_t) row * SYCL_உச்சி_K_RADIX_ROW_சொற்கள்,
                        (முழு) ncols, k, shift, is_first, part, nparts,
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
                மாறிலி முழு g    = item_ct1.get_group(0);
                மாறிலி முழு row  = g / nparts;
                மாறிலி முழு part = g % nparts;

                உச்சி_k_radix_split_emit_f32(
                    src + (int64_t) row * ncols,
                    dst_indices + (int64_t) row * k,
                    state + (int64_t) row * SYCL_உச்சி_K_RADIX_ROW_சொற்கள்,
                    (முழு) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item_ct1);
            });
    });
}

வெறுமை ggml_sycl_உச்சி_k_radix(
    ggml_backend_sycl_context & ctx,
    மாறிலி தசம *   src,
    int32_t *       dst_indices,
    மாறிலி int64_t   ncols,
    மாறிலி int64_t   nrows,
    மாறிலி முழு       k,
    dpct::queue_ptr main_stream
) {
    மாறிலி முழு nparts = உச்சி_k_radix_split_groups(ctx.device, ncols, nrows);
    எனில் (nparts > 1) {
        உச்சி_k_radix_split_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_stream);
    } இல்லையெனில் {
        உச்சி_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_stream);
    }
}
