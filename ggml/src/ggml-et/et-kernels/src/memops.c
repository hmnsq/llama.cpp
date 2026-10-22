//******************************************************************************
// Memory Operations Kernel — tensor_store based memset
//
// Uses the tensor engine's store path (bypasses L1+L2 caches) to achieve hiher
// performance. Unrolled vector writes can write at ~25GB/s and tensor writes
// can so ~71 GB/s. Only even harts (hart 0 per minion) participate, as due to
// hardware design (only thye have matrix engine access and co-op stores seems
// slower)
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// Operation identifiers for memops kernel
ਗਿਣਤੀ ggml_et_memop_type {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation parameters (must match host-side struct in ggml-et-memops.cpp)
ਬਣਤਰ memset_params {
    uint32_t op_type;
    uint32_t ਮੁੱਲ;
    ਖਾਲੀ *   dst_ptr;
    size_t   ਮਾਪ;
};

// Fill all 32 f-regs with a replicated byte pattern
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) fill_fregs(uint32_t fill32) {
    ਰਜਿਸਟਰ uint64_t val __asm__("a2") = fill32;
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
ਸਥਿਰ ਖਾਲੀ memset_ਪੂਛ(uint8_t * ਸ਼ੁਰੂ, uint8_t * ਸਿਰਾ, uint8_t val) {
    uint8_t * cur = ਸ਼ੁਰੂ;

    // Full 64-byte rows via tensor_store (up to 16 at a time = 1KB)
    ਜਦੋਂ (cur + 64 <= ਸਿਰਾ) {
        size_t ਪੰਕਤੀਆਂ = (ਸਿਰਾ - cur) / 64;
        ਜੇ (ਪੰਕਤੀਆਂ > 16) {
            ਪੰਕਤੀਆਂ = 16;
        }
        tensor_store(0, 0, 3, ਪੰਕਤੀਆਂ - 1, (uintptr_t) cur, 0, 64);
        cur += ਪੰਕਤੀਆਂ * 64;
    }

    // Remaining 16-byte aligned chunk (16, 32, or 48 bytes)
    ਜੇ (cur + 16 <= ਸਿਰਾ) {
        size_t ਖਾਨੇ_ਗਿਣਤੀ_2 = (ਸਿਰਾ - cur) / 16;
        tensor_store(0, 0, ਖਾਨੇ_ਗਿਣਤੀ_2 - 1, 0, (uintptr_t) cur, 0, 64);
        cur += ਖਾਨੇ_ਗਿਣਤੀ_2 * 16;
    }

    tensor_wait(TENSOR_STORE_WAIT);

    // Final < 16 bytes with byte stores
    ਜਦੋਂ (cur < ਸਿਰਾ) {
        *(ਅਸਥਿਰ uint8_t *) cur = val;
        cur++;
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ALIGN_ਉੱਪਰ(ptr, align) ((uint8_t *) (((uintptr_t) (ptr) + (align) - 1) & ~((uintptr_t) (align) - 1)))

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ memset_params * params, kernel_environment_t * env) {
    uint64_t hart_id = get_hart_id();

    // Only even harts have tensor engine access
    ਜੇ (hart_id & 1) {
        ਵਾਪਸ 0;
    }

    ਜੇ (!params || ((uintptr_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਜੇ (params->op_type != GGML_ET_MEMOP_MEMSET) {
        ਵਾਪਸ -1;
    }

    uint8_t * dst  = (uint8_t *) params->dst_ptr;
    size_t    ਮਾਪ = params->ਮਾਪ;

    ਜੇ (!dst || ਮਾਪ == 0) {
        ਵਾਪਸ -1;
    }

    // Dynamic hart count from shire_mask
    ਪੂਰਨ ਅੰਕ_even_harts = manual_popcountll(env->shire_ਨਕਾਬ) * SOC_MINIONS_ਪ੍ਰਤੀ_SHIRE;

    // global_id: shire * 32 + minion (for even harts)
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    uint8_t  val    = params->ਮੁੱਲ & 0xFF;
    uint32_t fill32 = val | ((uint32_t) val << 8) | ((uint32_t) val << 16) | ((uint32_t) val << 24);

    uint8_t * ਸਿਰਾ = dst + ਮਾਪ;

    ਤਿਆਰੀ_cache_scp();
    CLEAR_TENSOR_ERROR;
    fill_fregs(fill32);

    // Align to 16 bytes (tensor_store minimum alignment)
    uint8_t * ਅਧਾਰ = ALIGN_ਉੱਪਰ(dst, 16);
    ਜੇ (ਅਧਾਰ > ਸਿਰਾ) {
        ਅਧਾਰ = ਸਿਰਾ;
    }

    // Hart 0 handles head bytes before alignment
    ਜੇ (global_id == 0) {
        ਅਸਥਿਰ uint8_t * p = dst;
        ਜਦੋਂ (p < (ਅਸਥਿਰ uint8_t *) ਅਧਾਰ) {
            *p++ = val;
        }
    }

    // Bulk: 1KB blocks distributed across all harts (base is already 16-byte aligned)
    size_t aligned_ਮਾਪ = ਸਿਰਾ - ਅਧਾਰ;
    size_t ਕੁੱਲ_ਖੰਡਾਂ = aligned_ਮਾਪ / 1024;

    ਜੇ (ਕੁੱਲ_ਖੰਡਾਂ > 0) {
        size_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_hart = ਕੁੱਲ_ਖੰਡਾਂ / ਅੰਕ_even_harts;
        size_t extra           = ਕੁੱਲ_ਖੰਡਾਂ % ਅੰਕ_even_harts;
        size_t my_ਸ਼ੁਰੂ        = ਖੰਡਾਂ_ਪ੍ਰਤੀ_hart * global_id + (global_id < extra ? global_id : extra);
        size_t my_ਗਿਣਤੀ        = ਖੰਡਾਂ_ਪ੍ਰਤੀ_hart + (global_id < extra ? 1 : 0);

        uint8_t * addr = ਅਧਾਰ + my_ਸ਼ੁਰੂ * 1024;
        ਲਈ (size_t b = 0; b < my_ਗਿਣਤੀ; b++) {
            tensor_store(0, 0, 3, 15, (uintptr_t) addr, 0, 64);
            addr += 1024;
        }
        tensor_wait(TENSOR_STORE_WAIT);
    }

    // Hart 0 handles the tail after the last full 1KB block
    ਜੇ (global_id == 0) {
        memset_ਪੂਛ(ਅਧਾਰ + ਕੁੱਲ_ਖੰਡਾਂ * 1024, ਸਿਰਾ, val);
    }

    FENCE;
    ਵਾਪਸ 0;
}
