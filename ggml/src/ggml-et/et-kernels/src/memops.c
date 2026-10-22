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
    uint32_t मानम्;
    शून्य *   dst_ptr;
    size_t   आकार;
};

// Fill all 32 f-regs with a replicated byte pattern
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) fill_fregs(uint32_t fill32) {
    पञ्जिका uint64_t val __asm__("a2") = fill32;
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
स्थिर शून्य memset_पुच्छ(uint8_t * start, uint8_t * अन्तः_2, uint8_t val) {
    uint8_t * cur = start;

    // Full 64-byte rows via tensor_store (up to 16 at a time = 1KB)
    यावत् (cur + 64 <= अन्तः_2) {
        size_t पङ्क्तिकाः = (अन्तः_2 - cur) / 64;
        यदि (पङ्क्तिकाः > 16) {
            पङ्क्तिकाः = 16;
        }
        tensor_store(0, 0, 3, पङ्क्तिकाः - 1, (uintptr_t) cur, 0, 64);
        cur += पङ्क्तिकाः * 64;
    }

    // Remaining 16-byte aligned chunk (16, 32, or 48 bytes)
    यदि (cur + 16 <= अन्तः_2) {
        size_t स्तम्भसङ्ख्या_2 = (अन्तः_2 - cur) / 16;
        tensor_store(0, 0, स्तम्भसङ्ख्या_2 - 1, 0, (uintptr_t) cur, 0, 64);
        cur += स्तम्भसङ्ख्या_2 * 16;
    }

    tensor_wait(TENSOR_STORE_WAIT);

    // Final < 16 bytes with byte stores
    यावत् (cur < अन्तः_2) {
        *(चञ्चल uint8_t *) cur = val;
        cur++;
    }
}

#परिभाषय ALIGN_ऊर्ध्वम्(ptr, align) ((uint8_t *) (((uintptr_t) (ptr) + (align) - 1) & ~((uintptr_t) (align) - 1)))

पूर्णाङ्क प्रविष्टिः_point(संरचना memset_params * params, kernel_environment_t * env) {
    uint64_t hart_id = get_hart_id();

    // Only even harts have tensor engine access
    यदि (hart_id & 1) {
        प्रत्यागम 0;
    }

    यदि (!params || ((uintptr_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    यदि (params->op_type != GGML_ET_MEMOP_MEMSET) {
        प्रत्यागम -1;
    }

    uint8_t * dst  = (uint8_t *) params->dst_ptr;
    size_t    आकार = params->आकार;

    यदि (!dst || आकार == 0) {
        प्रत्यागम -1;
    }

    // Dynamic hart count from shire_mask
    पूर्णाङ्क अङ्कः_even_harts = manual_popcountll(env->shire_आवरणम्) * SOC_MINIONS_प्रत्येकम्_SHIRE;

    // global_id: shire * 32 + minion (for even harts)
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    uint8_t  val    = params->मानम् & 0xFF;
    uint32_t fill32 = val | ((uint32_t) val << 8) | ((uint32_t) val << 16) | ((uint32_t) val << 24);

    uint8_t * अन्तः_2 = dst + आकार;

    सज्जता_cache_scp();
    CLEAR_TENSOR_ERROR;
    fill_fregs(fill32);

    // Align to 16 bytes (tensor_store minimum alignment)
    uint8_t * आधारः = ALIGN_ऊर्ध्वम्(dst, 16);
    यदि (आधारः > अन्तः_2) {
        आधारः = अन्तः_2;
    }

    // Hart 0 handles head bytes before alignment
    यदि (global_id == 0) {
        चञ्चल uint8_t * p = dst;
        यावत् (p < (चञ्चल uint8_t *) आधारः) {
            *p++ = val;
        }
    }

    // Bulk: 1KB blocks distributed across all harts (base is already 16-byte aligned)
    size_t aligned_आकार = अन्तः_2 - आधारः;
    size_t योग_खण्डानि = aligned_आकार / 1024;

    यदि (योग_खण्डानि > 0) {
        size_t खण्डानि_प्रत्येकम्_hart = योग_खण्डानि / अङ्कः_even_harts;
        size_t extra           = योग_खण्डानि % अङ्कः_even_harts;
        size_t my_start        = खण्डानि_प्रत्येकम्_hart * global_id + (global_id < extra ? global_id : extra);
        size_t my_गणनम्        = खण्डानि_प्रत्येकम्_hart + (global_id < extra ? 1 : 0);

        uint8_t * addr = आधारः + my_start * 1024;
        प्रति (size_t b = 0; b < my_गणनम्; b++) {
            tensor_store(0, 0, 3, 15, (uintptr_t) addr, 0, 64);
            addr += 1024;
        }
        tensor_wait(TENSOR_STORE_WAIT);
    }

    // Hart 0 handles the tail after the last full 1KB block
    यदि (global_id == 0) {
        memset_पुच्छ(आधारः + योग_खण्डानि * 1024, अन्तः_2, val);
    }

    FENCE;
    प्रत्यागम 0;
}
