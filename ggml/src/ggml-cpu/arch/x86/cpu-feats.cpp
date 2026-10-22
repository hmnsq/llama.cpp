#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಯದಿ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#ಅನ್ತರ್ಭಾವ <intrin.h>
#endif

#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <bitset>
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ಸಂರಚನಾ cpuid_x86 {
    ತಾರ್ಕಿಕ SSE3(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[0]; }
    ತಾರ್ಕಿಕ PCLMULQDQ(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[1]; }
    ತಾರ್ಕಿಕ MONITOR(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[3]; }
    ತಾರ್ಕಿಕ SSSE3(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[9]; }
    ತಾರ್ಕಿಕ FMA(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[12]; }
    ತಾರ್ಕಿಕ CMPXCHG16B(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[13]; }
    ತಾರ್ಕಿಕ SSE41(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[19]; }
    ತಾರ್ಕಿಕ SSE42(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[20]; }
    ತಾರ್ಕಿಕ MOVBE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[22]; }
    ತಾರ್ಕಿಕ POPCNT(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[23]; }
    ತಾರ್ಕಿಕ AES(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[25]; }
    ತಾರ್ಕಿಕ XSAVE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[26]; }
    ತಾರ್ಕಿಕ OSXSAVE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[27]; }
    ತಾರ್ಕಿಕ AVX(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[28]; }
    ತಾರ್ಕಿಕ F16C(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[29]; }
    ತಾರ್ಕಿಕ RDRAND(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_ecx[30]; }

    ತಾರ್ಕಿಕ MSR(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[5]; }
    ತಾರ್ಕಿಕ CX8(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[8]; }
    ತಾರ್ಕಿಕ SEP(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[11]; }
    ತಾರ್ಕಿಕ CMOV(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[15]; }
    ತಾರ್ಕಿಕ CLFSH(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[19]; }
    ತಾರ್ಕಿಕ MMX(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[23]; }
    ತಾರ್ಕಿಕ FXSR(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[24]; }
    ತಾರ್ಕಿಕ SSE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[25]; }
    ತಾರ್ಕಿಕ SSE2(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_1_edx[26]; }

    ತಾರ್ಕಿಕ FSGSBASE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[0]; }
    ತಾರ್ಕಿಕ BMI1(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[3]; }
    ತಾರ್ಕಿಕ HLE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_intel && f_7_ebx[4]; }
    ತಾರ್ಕಿಕ AVX2(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[5]; }
    ತಾರ್ಕಿಕ BMI2(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[8]; }
    ತಾರ್ಕಿಕ ERMS(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[9]; }
    ತಾರ್ಕಿಕ INVPCID(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[10]; }
    ತಾರ್ಕಿಕ RTM(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_intel && f_7_ebx[11]; }
    ತಾರ್ಕಿಕ AVX512F(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[16]; }
    ತಾರ್ಕಿಕ AVX512DQ(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[17]; }
    ತಾರ್ಕಿಕ RDSEED(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[18]; }
    ತಾರ್ಕಿಕ ADX(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[19]; }
    ತಾರ್ಕಿಕ AVX512PF(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[26]; }
    ತಾರ್ಕಿಕ AVX512ER(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[27]; }
    ತಾರ್ಕಿಕ AVX512CD(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[28]; }
    ತಾರ್ಕಿಕ AVX512BW(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[30]; }
    ತಾರ್ಕಿಕ AVX512VL(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[31]; }

    ತಾರ್ಕಿಕ SHA(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ebx[29]; }

    ತಾರ್ಕಿಕ PREFETCHWT1(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ecx[0]; }

    ತಾರ್ಕಿಕ LAHF(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_81_ecx[0]; }
    ತಾರ್ಕಿಕ LZCNT(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_intel && f_81_ecx[5]; }
    ತಾರ್ಕಿಕ ABM(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_ecx[5]; }
    ತಾರ್ಕಿಕ SSE4a(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_ecx[6]; }
    ತಾರ್ಕಿಕ XOP(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_ecx[11]; }
    ತಾರ್ಕಿಕ TBM(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_ecx[21]; }

    ತಾರ್ಕಿಕ SYSCALL(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_intel && f_81_edx[11]; }
    ತಾರ್ಕಿಕ MMXEXT(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_edx[22]; }
    ತಾರ್ಕಿಕ RDTSCP(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_intel && f_81_edx[27]; }
    ತಾರ್ಕಿಕ _3DNOWEXT(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_edx[30]; }
    ತಾರ್ಕಿಕ _3DNOW(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ is_amd && f_81_edx[31]; }

    ತಾರ್ಕಿಕ AVX512_VBMI(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ecx[1]; }
    ತಾರ್ಕಿಕ AVX512_VNNI(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_ecx[11]; }
    ತಾರ್ಕಿಕ AVX512_FP16(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_edx[23]; }
    ತಾರ್ಕಿಕ AVX512_BF16(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_1_eax[5]; }
    ತಾರ್ಕಿಕ AVX_VNNI(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_1_eax[4]; }

    ತಾರ್ಕಿಕ AMX_TILE(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_edx[24]; }
    ತಾರ್ಕಿಕ AMX_INT8(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_edx[25]; }
    ತಾರ್ಕಿಕ AMX_FP16(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_1_eax[21]; }
    ತಾರ್ಕಿಕ AMX_BF16(ಶೂನ್ಯ) { ಪ್ರತ್ಯಾಗಮ f_7_edx[22]; }

#ifdef _MSC_VER
    ಸ್ಥಿರ ಶೂನ್ಯ cpuid(ಪೂರ್ಣಾಙ್ಕ cpu_info[4], ಪೂರ್ಣಾಙ್ಕ eax) {
        __cpuid(cpu_info, eax);
    }
    ಸ್ಥಿರ ಶೂನ್ಯ cpuidex(ಪೂರ್ಣಾಙ್ಕ cpu_info[4], ಪೂರ್ಣಾಙ್ಕ eax, ಪೂರ್ಣಾಙ್ಕ ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ಅನ್ಯಥಾ
    ಸ್ಥಿರ ಶೂನ್ಯ cpuid(ಪೂರ್ಣಾಙ್ಕ cpu_info[4], ಪೂರ್ಣಾಙ್ಕ eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ಸ್ಥಿರ ಶೂನ್ಯ cpuidex(ಪೂರ್ಣಾಙ್ಕ cpu_info[4], ಪೂರ್ಣಾಙ್ಕ eax, ಪೂರ್ಣಾಙ್ಕ ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<ಪೂರ್ಣಾಙ್ಕ, 4> cpui;
        std::vector<std::array<ಪೂರ್ಣಾಙ್ಕ, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        ಪೂರ್ಣಾಙ್ಕ n_ids = cpui[0];

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        ವರ್ಣ vendor[0x20] = {};
        *reinterpret_cast<ಪೂರ್ಣಾಙ್ಕ *>(vendor)     = data[0][1];
        *reinterpret_cast<ಪೂರ್ಣಾಙ್ಕ *>(vendor + 4) = data[0][3];
        *reinterpret_cast<ಪೂರ್ಣಾಙ್ಕ *>(vendor + 8) = data[0][2];
        ಏತತ್->vendor = vendor;
        ಯದಿ (ಏತತ್->vendor == "GenuineIntel") {
            is_intel = ಸತ್ಯ;
        } ಅನ್ಯಥಾ ಯದಿ (ಏತತ್->vendor == "AuthenticAMD") {
            is_amd = ಸತ್ಯ;
        }

        // load bitset with flags for function 0x00000001
        ಯದಿ (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        ಯದಿ (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned ಪೂರ್ಣಾಙ್ಕ n_ex_ids = cpui[0];

        std::vector<std::array<ಪೂರ್ಣಾಙ್ಕ, 4>> ext_data;
        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ಯದಿ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        ವರ್ಣ brand[0x40] = {};
        ಯದಿ (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), ಪರಿಮಾಣ(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), ಪರಿಮಾಣ(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), ಪರಿಮಾಣ(cpui));
            ಏತತ್->brand = brand;
        }
    }

    ತಾರ್ಕಿಕ is_intel = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ is_amd = ಅಸತ್ಯ;
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

#ಯದಿ 0
ಶೂನ್ಯ test_x86_is() {
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

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    ಪೂರ್ಣಾಙ್ಕ score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    ಯದಿ (!is.FMA()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    ಯದಿ (!is.F16C()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    ಯದಿ (!is.SSE42()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    ಯದಿ (!is.BMI2()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    ಯದಿ (!is.AVX()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    ಯದಿ (!is.AVX2()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    ಯದಿ (!is.AVX_VNNI()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    ಯದಿ (!is.AVX512F()) { ಪ್ರತ್ಯಾಗಮ 0; }
    ಯದಿ (!is.AVX512CD()) { ಪ್ರತ್ಯಾಗಮ 0; }
    ಯದಿ (!is.AVX512VL()) { ಪ್ರತ್ಯಾಗಮ 0; }
    ಯದಿ (!is.AVX512DQ()) { ಪ್ರತ್ಯಾಗಮ 0; }
    ಯದಿ (!is.AVX512BW()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    ಯದಿ (!is.AVX512_VBMI()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    ಯದಿ (!is.AVX512_BF16()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    ಯದಿ (!is.AVX512_VNNI()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    ಯದಿ (!is.AMX_INT8()) { ಪ್ರತ್ಯಾಗಮ 0; }
    score += 1<<11;
#endif

    ಪ್ರತ್ಯಾಗಮ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
