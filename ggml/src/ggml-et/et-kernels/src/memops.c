//******************************************************************************
// Memory Operations Kernel — tensor_store based memset
//
// Uses the tensor engine's store path (bypasses L1+L2 caches) to achieve hiher
// performance. Unrolled vector writes can write at ~25GB/s and tensor writes
// can so ~71 GB/s. Only even harts (hart 0 per minion) participate, as due to
// hardware design (only thye have matrix engine access and co-op stores seems
// slower)
//******************************************************************************

#شامل "platform.h"
#شامل "tensor.h"

#شامل <etsoc/common/utils.h>
#شامل <stdint.h>

// Operation identifiers for memops kernel
شمار ggml_et_memop_type {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation parameters (must match host-side struct in ggml-et-memops.cpp)
ساخت memset_params {
    uint32_t op_type;
    uint32_t قدر;
    خالی *   dst_ptr;
    size_t   سائز;
};

// Fill all 32 f-regs with a replicated byte pattern
ساکن اندرونی خالی __attribute__((always_inline)) fill_fregs(uint32_t fill32) {
    اندراج uint64_t val __asm__("a2") = fill32;
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
ساکن خالی memset_دم(uint8_t * آغاز, uint8_t * انتہا, uint8_t val) {
    uint8_t * cur = آغاز;

    // Full 64-byte rows via tensor_store (up to 16 at a time = 1KB)
    جب (cur + 64 <= انتہا) {
        size_t صفیں = (انتہا - cur) / 64;
        اگر (صفیں > 16) {
            صفیں = 16;
        }
        tensor_store(0, 0, 3, صفیں - 1, (uintptr_t) cur, 0, 64);
        cur += صفیں * 64;
    }

    // Remaining 16-byte aligned chunk (16, 32, or 48 bytes)
    اگر (cur + 16 <= انتہا) {
        size_t ستون_تعداد_2 = (انتہا - cur) / 16;
        tensor_store(0, 0, ستون_تعداد_2 - 1, 0, (uintptr_t) cur, 0, 64);
        cur += ستون_تعداد_2 * 16;
    }

    tensor_wait(TENSOR_STORE_WAIT);

    // Final < 16 bytes with byte stores
    جب (cur < انتہا) {
        *(متلون uint8_t *) cur = val;
        cur++;
    }
}

#تعریف_کرو ALIGN_اوپر(ptr, align) ((uint8_t *) (((uintptr_t) (ptr) + (align) - 1) & ~((uintptr_t) (align) - 1)))

صحیح اندراج_نقطہ(ساخت memset_params * params, kernel_environment_t * env) {
    uint64_t hart_id = get_hart_id();

    // Only even harts have tensor engine access
    اگر (hart_id & 1) {
        واپس 0;
    }

    اگر (!params || ((uintptr_t) params & 0x7) != 0) {
        واپس -1;
    }

    اگر (params->op_type != GGML_ET_MEMOP_MEMSET) {
        واپس -1;
    }

    uint8_t * dst  = (uint8_t *) params->dst_ptr;
    size_t    سائز = params->سائز;

    اگر (!dst || سائز == 0) {
        واپس -1;
    }

    // Dynamic hart count from shire_mask
    صحیح عدد_even_harts = manual_popcountll(env->shire_نقاب) * SOC_MINIONS_فی_SHIRE;

    // global_id: shire * 32 + minion (for even harts)
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    uint8_t  val    = params->قدر & 0xFF;
    uint32_t fill32 = val | ((uint32_t) val << 8) | ((uint32_t) val << 16) | ((uint32_t) val << 24);

    uint8_t * انتہا = dst + سائز;

    تیاری_cache_scp();
    CLEAR_TENSOR_ERROR;
    fill_fregs(fill32);

    // Align to 16 bytes (tensor_store minimum alignment)
    uint8_t * بنیاد = ALIGN_اوپر(dst, 16);
    اگر (بنیاد > انتہا) {
        بنیاد = انتہا;
    }

    // Hart 0 handles head bytes before alignment
    اگر (global_id == 0) {
        متلون uint8_t * p = dst;
        جب (p < (متلون uint8_t *) بنیاد) {
            *p++ = val;
        }
    }

    // Bulk: 1KB blocks distributed across all harts (base is already 16-byte aligned)
    size_t aligned_سائز = انتہا - بنیاد;
    size_t کل_بلاکیں = aligned_سائز / 1024;

    اگر (کل_بلاکیں > 0) {
        size_t بلاکیں_فی_hart = کل_بلاکیں / عدد_even_harts;
        size_t extra           = کل_بلاکیں % عدد_even_harts;
        size_t my_آغاز        = بلاکیں_فی_hart * global_id + (global_id < extra ? global_id : extra);
        size_t my_گنتی        = بلاکیں_فی_hart + (global_id < extra ? 1 : 0);

        uint8_t * addr = بنیاد + my_آغاز * 1024;
        برائے (size_t b = 0; b < my_گنتی; b++) {
            tensor_store(0, 0, 3, 15, (uintptr_t) addr, 0, 64);
            addr += 1024;
        }
        tensor_wait(TENSOR_STORE_WAIT);
    }

    // Hart 0 handles the tail after the last full 1KB block
    اگر (global_id == 0) {
        memset_دم(بنیاد + کل_بلاکیں * 1024, انتہا, val);
    }

    FENCE;
    واپس 0;
}
