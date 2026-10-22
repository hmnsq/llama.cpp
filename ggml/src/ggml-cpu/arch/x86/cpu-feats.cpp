#অন্তর্ভাব "ggml-backend-impl.h"

#যদি defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#অন্তর্ভাব <intrin.h>
#endif

#অন্তর্ভাব <cstring>
#অন্তর্ভাব <vector>
#অন্তর্ভাব <bitset>
#অন্তর্ভাব <array>
#অন্তর্ভাব <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
কাঠামো cpuid_x86 {
    যৌক্তিক SSE3(শূন্য) { ফেরত f_1_ecx[0]; }
    যৌক্তিক PCLMULQDQ(শূন্য) { ফেরত f_1_ecx[1]; }
    যৌক্তিক MONITOR(শূন্য) { ফেরত f_1_ecx[3]; }
    যৌক্তিক SSSE3(শূন্য) { ফেরত f_1_ecx[9]; }
    যৌক্তিক FMA(শূন্য) { ফেরত f_1_ecx[12]; }
    যৌক্তিক CMPXCHG16B(শূন্য) { ফেরত f_1_ecx[13]; }
    যৌক্তিক SSE41(শূন্য) { ফেরত f_1_ecx[19]; }
    যৌক্তিক SSE42(শূন্য) { ফেরত f_1_ecx[20]; }
    যৌক্তিক MOVBE(শূন্য) { ফেরত f_1_ecx[22]; }
    যৌক্তিক POPCNT(শূন্য) { ফেরত f_1_ecx[23]; }
    যৌক্তিক AES(শূন্য) { ফেরত f_1_ecx[25]; }
    যৌক্তিক XSAVE(শূন্য) { ফেরত f_1_ecx[26]; }
    যৌক্তিক OSXSAVE(শূন্য) { ফেরত f_1_ecx[27]; }
    যৌক্তিক AVX(শূন্য) { ফেরত f_1_ecx[28]; }
    যৌক্তিক F16C(শূন্য) { ফেরত f_1_ecx[29]; }
    যৌক্তিক RDRAND(শূন্য) { ফেরত f_1_ecx[30]; }

    যৌক্তিক MSR(শূন্য) { ফেরত f_1_edx[5]; }
    যৌক্তিক CX8(শূন্য) { ফেরত f_1_edx[8]; }
    যৌক্তিক SEP(শূন্য) { ফেরত f_1_edx[11]; }
    যৌক্তিক CMOV(শূন্য) { ফেরত f_1_edx[15]; }
    যৌক্তিক CLFSH(শূন্য) { ফেরত f_1_edx[19]; }
    যৌক্তিক MMX(শূন্য) { ফেরত f_1_edx[23]; }
    যৌক্তিক FXSR(শূন্য) { ফেরত f_1_edx[24]; }
    যৌক্তিক SSE(শূন্য) { ফেরত f_1_edx[25]; }
    যৌক্তিক SSE2(শূন্য) { ফেরত f_1_edx[26]; }

    যৌক্তিক FSGSBASE(শূন্য) { ফেরত f_7_ebx[0]; }
    যৌক্তিক BMI1(শূন্য) { ফেরত f_7_ebx[3]; }
    যৌক্তিক HLE(শূন্য) { ফেরত is_intel && f_7_ebx[4]; }
    যৌক্তিক AVX2(শূন্য) { ফেরত f_7_ebx[5]; }
    যৌক্তিক BMI2(শূন্য) { ফেরত f_7_ebx[8]; }
    যৌক্তিক ERMS(শূন্য) { ফেরত f_7_ebx[9]; }
    যৌক্তিক INVPCID(শূন্য) { ফেরত f_7_ebx[10]; }
    যৌক্তিক RTM(শূন্য) { ফেরত is_intel && f_7_ebx[11]; }
    যৌক্তিক AVX512F(শূন্য) { ফেরত f_7_ebx[16]; }
    যৌক্তিক AVX512DQ(শূন্য) { ফেরত f_7_ebx[17]; }
    যৌক্তিক RDSEED(শূন্য) { ফেরত f_7_ebx[18]; }
    যৌক্তিক ADX(শূন্য) { ফেরত f_7_ebx[19]; }
    যৌক্তিক AVX512PF(শূন্য) { ফেরত f_7_ebx[26]; }
    যৌক্তিক AVX512ER(শূন্য) { ফেরত f_7_ebx[27]; }
    যৌক্তিক AVX512CD(শূন্য) { ফেরত f_7_ebx[28]; }
    যৌক্তিক AVX512BW(শূন্য) { ফেরত f_7_ebx[30]; }
    যৌক্তিক AVX512VL(শূন্য) { ফেরত f_7_ebx[31]; }

    যৌক্তিক SHA(শূন্য) { ফেরত f_7_ebx[29]; }

    যৌক্তিক PREFETCHWT1(শূন্য) { ফেরত f_7_ecx[0]; }

    যৌক্তিক LAHF(শূন্য) { ফেরত f_81_ecx[0]; }
    যৌক্তিক LZCNT(শূন্য) { ফেরত is_intel && f_81_ecx[5]; }
    যৌক্তিক ABM(শূন্য) { ফেরত is_amd && f_81_ecx[5]; }
    যৌক্তিক SSE4a(শূন্য) { ফেরত is_amd && f_81_ecx[6]; }
    যৌক্তিক XOP(শূন্য) { ফেরত is_amd && f_81_ecx[11]; }
    যৌক্তিক TBM(শূন্য) { ফেরত is_amd && f_81_ecx[21]; }

    যৌক্তিক SYSCALL(শূন্য) { ফেরত is_intel && f_81_edx[11]; }
    যৌক্তিক MMXEXT(শূন্য) { ফেরত is_amd && f_81_edx[22]; }
    যৌক্তিক RDTSCP(শূন্য) { ফেরত is_intel && f_81_edx[27]; }
    যৌক্তিক _3DNOWEXT(শূন্য) { ফেরত is_amd && f_81_edx[30]; }
    যৌক্তিক _3DNOW(শূন্য) { ফেরত is_amd && f_81_edx[31]; }

    যৌক্তিক AVX512_VBMI(শূন্য) { ফেরত f_7_ecx[1]; }
    যৌক্তিক AVX512_VNNI(শূন্য) { ফেরত f_7_ecx[11]; }
    যৌক্তিক AVX512_FP16(শূন্য) { ফেরত f_7_edx[23]; }
    যৌক্তিক AVX512_BF16(শূন্য) { ফেরত f_7_1_eax[5]; }
    যৌক্তিক AVX_VNNI(শূন্য) { ফেরত f_7_1_eax[4]; }

    যৌক্তিক AMX_TILE(শূন্য) { ফেরত f_7_edx[24]; }
    যৌক্তিক AMX_INT8(শূন্য) { ফেরত f_7_edx[25]; }
    যৌক্তিক AMX_FP16(শূন্য) { ফেরত f_7_1_eax[21]; }
    যৌক্তিক AMX_BF16(শূন্য) { ফেরত f_7_edx[22]; }

#ifdef _MSC_VER
    স্থির শূন্য cpuid(পূর্ণসংখ্যা cpu_info[4], পূর্ণসংখ্যা eax) {
        __cpuid(cpu_info, eax);
    }
    স্থির শূন্য cpuidex(পূর্ণসংখ্যা cpu_info[4], পূর্ণসংখ্যা eax, পূর্ণসংখ্যা ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#অন্যথা
    স্থির শূন্য cpuid(পূর্ণসংখ্যা cpu_info[4], পূর্ণসংখ্যা eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    স্থির শূন্য cpuidex(পূর্ণসংখ্যা cpu_info[4], পূর্ণসংখ্যা eax, পূর্ণসংখ্যা ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<পূর্ণসংখ্যা, 4> cpui;
        std::vector<std::array<পূর্ণসংখ্যা, 4>> তথ্য;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.তথ্য(), 0);
        পূর্ণসংখ্যা n_ids = cpui[0];

        জন্য (পূর্ণসংখ্যা i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.তথ্য(), i, 0);
            তথ্য.push_back(cpui);
        }

        // capture vendor string
        অক্ষর vendor[0x20] = {};
        *reinterpret_cast<পূর্ণসংখ্যা *>(vendor)     = তথ্য[0][1];
        *reinterpret_cast<পূর্ণসংখ্যা *>(vendor + 4) = তথ্য[0][3];
        *reinterpret_cast<পূর্ণসংখ্যা *>(vendor + 8) = তথ্য[0][2];
        এই->vendor = vendor;
        যদি (এই->vendor == "GenuineIntel") {
            is_intel = সত্য;
        } অন্যথা যদি (এই->vendor == "AuthenticAMD") {
            is_amd = সত্য;
        }

        // load bitset with flags for function 0x00000001
        যদি (n_ids >= 1) {
            f_1_ecx = তথ্য[1][2];
            f_1_edx = তথ্য[1][3];
        }

        // load bitset with flags for function 0x00000007
        যদি (n_ids >= 7) {
            f_7_ebx = তথ্য[7][1];
            f_7_ecx = তথ্য[7][2];
            f_7_edx = তথ্য[7][3];
            cpuidex(cpui.তথ্য(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.তথ্য(), 0x80000000);
        unsigned পূর্ণসংখ্যা n_ex_ids = cpui[0];

        std::vector<std::array<পূর্ণসংখ্যা, 4>> ext_তথ্য;
        জন্য (unsigned পূর্ণসংখ্যা i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.তথ্য(), i, 0);
            ext_তথ্য.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        যদি (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_তথ্য[1][2];
            f_81_edx = ext_তথ্য[1][3];
        }

        // interpret CPU brand string if reported
        অক্ষর brand[0x40] = {};
        যদি (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_তথ্য[2].তথ্য(), আকার(cpui));
            std::memcpy(brand + 16, ext_তথ্য[3].তথ্য(), আকার(cpui));
            std::memcpy(brand + 32, ext_তথ্য[4].তথ্য(), আকার(cpui));
            এই->brand = brand;
        }
    }

    যৌক্তিক is_intel = মিথ্যা;
    যৌক্তিক is_amd = মিথ্যা;
    std::string vendor;
    std::string brand;
    std::bitset<32> f_1_ecx;
    std::bitset<32> f_1_edx;
    std::bitset<32> f_7_ebx;
    std::bitset<32> f_7_ecx;
    std::bitset<32> f_7_edx;
    std::bitset<32> f_7_1_eax;
    std::bitset<32> f_81_ecx;
    std::bitset<32> f_81_edx;
};

#যদি 0
শূন্য test_x86_is() {
    cpuid_x86 is;
    printf("CPU Vendor: %s\n", is.vendor.c_str());
    printf("Brand: %s\n", is.brand.c_str());
    printf("is_intel: %d\n", is.is_intel);
    printf("is_amd: %d\n", is.is_amd);
    printf("sse3: %d\n", is.SSE3());
    printf("pclmulqdq: %d\n", is.PCLMULQDQ());
    printf("ssse3: %d\n", is.SSSE3());
    printf("fma: %d\n", is.FMA());
    printf("cmpxchg16b: %d\n", is.CMPXCHG16B());
    printf("sse41: %d\n", is.SSE41());
    printf("sse42: %d\n", is.SSE42());
    printf("movbe: %d\n", is.MOVBE());
    printf("popcnt: %d\n", is.POPCNT());
    printf("aes: %d\n", is.AES());
    printf("xsave: %d\n", is.XSAVE());
    printf("osxsave: %d\n", is.OSXSAVE());
    printf("avx: %d\n", is.AVX());
    printf("f16c: %d\n", is.F16C());
    printf("rdrand: %d\n", is.RDRAND());
    printf("msr: %d\n", is.MSR());
    printf("cx8: %d\n", is.CX8());
    printf("sep: %d\n", is.SEP());
    printf("cmov: %d\n", is.CMOV());
    printf("clflush: %d\n", is.CLFSH());
    printf("mmx: %d\n", is.MMX());
    printf("fxsr: %d\n", is.FXSR());
    printf("sse: %d\n", is.SSE());
    printf("sse2: %d\n", is.SSE2());
    printf("fsgsbase: %d\n", is.FSGSBASE());
    printf("bmi1: %d\n", is.BMI1());
    printf("hle: %d\n", is.HLE());
    printf("avx2: %d\n", is.AVX2());
    printf("bmi2: %d\n", is.BMI2());
    printf("erms: %d\n", is.ERMS());
    printf("invpcid: %d\n", is.INVPCID());
    printf("rtm: %d\n", is.RTM());
    printf("avx512f: %d\n", is.AVX512F());
    printf("rdseed: %d\n", is.RDSEED());
    printf("adx: %d\n", is.ADX());
    printf("avx512pf: %d\n", is.AVX512PF());
    printf("avx512er: %d\n", is.AVX512ER());
    printf("avx512cd: %d\n", is.AVX512CD());
    printf("sha: %d\n", is.SHA());
    printf("prefetchwt1: %d\n", is.PREFETCHWT1());
    printf("lahf: %d\n", is.LAHF());
    printf("lzcnt: %d\n", is.LZCNT());
    printf("abm: %d\n", is.ABM());
    printf("sse4a: %d\n", is.SSE4a());
    printf("xop: %d\n", is.XOP());
    printf("tbm: %d\n", is.TBM());
    printf("syscall: %d\n", is.SYSCALL());
    printf("mmxext: %d\n", is.MMXEXT());
    printf("rdtscp: %d\n", is.RDTSCP());
    printf("3dnowext: %d\n", is._3DNOWEXT());
    printf("3dnow: %d\n", is._3DNOW());
    printf("avx512_vbmi: %d\n", is.AVX512_VBMI());
    printf("avx512_vnni: %d\n", is.AVX512_VNNI());
    printf("avx512_fp16: %d\n", is.AVX512_FP16());
    printf("avx512_bf16: %d\n", is.AVX512_BF16());
    printf("amx_tile: %d\n", is.AMX_TILE());
    printf("amx_int8: %d\n", is.AMX_INT8());
    printf("amx_fp16: %d\n", is.AMX_FP16());
    printf("amx_bf16: %d\n", is.AMX_BF16());
}
#endif

স্থির পূর্ণসংখ্যা ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    পূর্ণসংখ্যা score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    যদি (!is.FMA()) { ফেরত 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    যদি (!is.F16C()) { ফেরত 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    যদি (!is.SSE42()) { ফেরত 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    যদি (!is.BMI2()) { ফেরত 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    যদি (!is.AVX()) { ফেরত 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    যদি (!is.AVX2()) { ফেরত 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    যদি (!is.AVX_VNNI()) { ফেরত 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    যদি (!is.AVX512F()) { ফেরত 0; }
    যদি (!is.AVX512CD()) { ফেরত 0; }
    যদি (!is.AVX512VL()) { ফেরত 0; }
    যদি (!is.AVX512DQ()) { ফেরত 0; }
    যদি (!is.AVX512BW()) { ফেরত 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    যদি (!is.AVX512_VBMI()) { ফেরত 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    যদি (!is.AVX512_BF16()) { ফেরত 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    যদি (!is.AVX512_VNNI()) { ফেরত 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    যদি (!is.AMX_INT8()) { ফেরত 0; }
    score += 1<<11;
#endif

    ফেরত score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
