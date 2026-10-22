//******************************************************************************
// Memory Operations Kernel — tensor_store based memset
//
// Uses the tensor engine's store path (bypasses L1+L2 caches) to achieve hiher
// performance. Unrolled vector writes can write at ~25GB/s and tensor writes
// can so ~71 GB/s. Only even harts (hart 0 per minion) participate, as due to
// hardware design (only thye have matrix engine access and co-op stores seems
// slower)
//******************************************************************************

#अन्तर्भाव "platform.h"
#अन्तर्भाव "tensor.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>

// Operation identifiers for memops kernel
गणना ggml_et_memop_type {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation parameters (must match host-side struct in ggml-et-memops.cpp)
संरचना memset_params {
    uint32_t op_type;
    uint32_t मूल्य;
    शून्य *   dst_ptr;
    माप_प्रकार   size;
};

// Fill all 32 f-regs with a replicated byte pattern
स्थिर ओळीत शून्य __attribute__((always_inline)) fill_fregs(uint32_t fill32) {
    नोंदणी uint64_t val __asm__("a2") = fill32;
    __asm__ __volatile__(
        "fbcx.ps f0, %[v]\n\t"
        "fbcx.ps f1, %[v]\n\t"
        "fbcx.ps f2, %[v]\n\t"
        "fbcx.ps f3, %[v]\n\t"
        "fbcx.ps f4, %[v]\n\t"
        "fbcx.ps f5, %[v]\n\t"
        "fbcx.ps f6, %[v]\n\t"
        "fbcx.ps f7, %[v]\n\t"
        "fbcx.ps f8, %[v]\n\t"
        "fbcx.ps f9, %[v]\n\t"
        "fbcx.ps f10, %[v]\n\t"
        "fbcx.ps f11, %[v]\n\t"
        "fbcx.ps f12, %[v]\n\t"
        "fbcx.ps f13, %[v]\n\t"
        "fbcx.ps f14, %[v]\n\t"
        "fbcx.ps f15, %[v]\n\t"
        "fbcx.ps f16, %[v]\n\t"
        "fbcx.ps f17, %[v]\n\t"
        "fbcx.ps f18, %[v]\n\t"
        "fbcx.ps f19, %[v]\n\t"
        "fbcx.ps f20, %[v]\n\t"
        "fbcx.ps f21, %[v]\n\t"
        "fbcx.ps f22, %[v]\n\t"
        "fbcx.ps f23, %[v]\n\t"
        "fbcx.ps f24, %[v]\n\t"
        "fbcx.ps f25, %[v]\n\t"
        "fbcx.ps f26, %[v]\n\t"
        "fbcx.ps f27, %[v]\n\t"
        "fbcx.ps f28, %[v]\n\t"
        "fbcx.ps f29, %[v]\n\t"
        "fbcx.ps f30, %[v]\n\t"
        "fbcx.ps f31, %[v]\n\t" ::[v] "r"(val)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
          "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
}

// Fill a partial region [start, end) using tensor_store for 16-byte-aligned
// chunks and byte stores for any remainder < 16 bytes.
// Assumes f-regs are already loaded with the fill pattern.
स्थिर शून्य memset_tail(uint8_t * start, uint8_t * end, uint8_t val) {
    uint8_t * cur = start;

    // Full 64-byte rows via tensor_store (up to 16 at a time = 1KB)
    पर्यंत (cur + 64 <= end) {
        माप_प्रकार rows = (end - cur) / 64;
        जर (rows > 16) {
            rows = 16;
        }
        tensor_store(0, 0, 3, rows - 1, (uintptr_t) cur, 0, 64);
        cur += rows * 64;
    }

    // Remaining 16-byte aligned chunk (16, 32, or 48 bytes)
    जर (cur + 16 <= end) {
        माप_प्रकार cols = (end - cur) / 16;
        tensor_store(0, 0, cols - 1, 0, (uintptr_t) cur, 0, 64);
        cur += cols * 16;
    }

    tensor_wait(TENSOR_STORE_WAIT);

    // Final < 16 bytes with byte stores
    पर्यंत (cur < end) {
        *(अस्थिर uint8_t *) cur = val;
        cur++;
    }
}

#define ALIGN_UP(ptr, align) ((uint8_t *) (((uintptr_t) (ptr) + (align) - 1) & ~((uintptr_t) (align) - 1)))

पूर्णांक entry_point(संरचना memset_params * params, kernel_environment_t * env) {
    uint64_t hart_id = get_hart_id();

    // Only even harts have tensor engine access
    जर (hart_id & 1) {
        परत 0;
    }

    जर (!params || ((uintptr_t) params & 0x7) != 0) {
        परत -1;
    }

    जर (params->op_type != GGML_ET_MEMOP_MEMSET) {
        परत -1;
    }

    uint8_t * dst  = (uint8_t *) params->dst_ptr;
    माप_प्रकार    size = params->size;

    जर (!dst || size == 0) {
        परत -1;
    }

    // Dynamic hart count from shire_mask
    पूर्णांक अंक_even_harts = manual_popcountll(env->shire_mask) * SOC_MINIONS_PER_SHIRE;

    // global_id: shire * 32 + minion (for even harts)
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    uint8_t  val    = params->मूल्य & 0xFF;
    uint32_t fill32 = val | ((uint32_t) val << 8) | ((uint32_t) val << 16) | ((uint32_t) val << 24);

    uint8_t * end = dst + size;

    setup_cache_scp();
    CLEAR_TENSOR_ERROR;
    fill_fregs(fill32);

    // Align to 16 bytes (tensor_store minimum alignment)
    uint8_t * base = ALIGN_UP(dst, 16);
    जर (base > end) {
        base = end;
    }

    // Hart 0 handles head bytes before alignment
    जर (global_id == 0) {
        अस्थिर uint8_t * p = dst;
        पर्यंत (p < (अस्थिर uint8_t *) base) {
            *p++ = val;
        }
    }

    // Bulk: 1KB blocks distributed across all harts (base is already 16-byte aligned)
    माप_प्रकार aligned_size = end - base;
    माप_प्रकार total_blocks = aligned_size / 1024;

    जर (total_blocks > 0) {
        माप_प्रकार blocks_per_hart = total_blocks / अंक_even_harts;
        माप_प्रकार extra           = total_blocks % अंक_even_harts;
        माप_प्रकार my_start        = blocks_per_hart * global_id + (global_id < extra ? global_id : extra);
        माप_प्रकार my_मोजणी        = blocks_per_hart + (global_id < extra ? 1 : 0);

        uint8_t * addr = base + my_start * 1024;
        साठी (माप_प्रकार b = 0; b < my_मोजणी; b++) {
            tensor_store(0, 0, 3, 15, (uintptr_t) addr, 0, 64);
            addr += 1024;
        }
        tensor_wait(TENSOR_STORE_WAIT);
    }

    // Hart 0 handles the tail after the last full 1KB block
    जर (global_id == 0) {
        memset_tail(base + total_blocks * 1024, end, val);
    }

    FENCE;
    परत 0;
}
