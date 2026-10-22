#അന്തര്ഭാവ "ggml-backend-impl.h"

#എങ്കിൽ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#അന്തര്ഭാവ <intrin.h>
#endif

#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <bitset>
#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ഘടന cpuid_x86 {
    തർക്കം SSE3(ശൂന്യം) { തിരികെ f_1_ecx[0]; }
    തർക്കം PCLMULQDQ(ശൂന്യം) { തിരികെ f_1_ecx[1]; }
    തർക്കം MONITOR(ശൂന്യം) { തിരികെ f_1_ecx[3]; }
    തർക്കം SSSE3(ശൂന്യം) { തിരികെ f_1_ecx[9]; }
    തർക്കം FMA(ശൂന്യം) { തിരികെ f_1_ecx[12]; }
    തർക്കം CMPXCHG16B(ശൂന്യം) { തിരികെ f_1_ecx[13]; }
    തർക്കം SSE41(ശൂന്യം) { തിരികെ f_1_ecx[19]; }
    തർക്കം SSE42(ശൂന്യം) { തിരികെ f_1_ecx[20]; }
    തർക്കം MOVBE(ശൂന്യം) { തിരികെ f_1_ecx[22]; }
    തർക്കം POPCNT(ശൂന്യം) { തിരികെ f_1_ecx[23]; }
    തർക്കം AES(ശൂന്യം) { തിരികെ f_1_ecx[25]; }
    തർക്കം XSAVE(ശൂന്യം) { തിരികെ f_1_ecx[26]; }
    തർക്കം OSXSAVE(ശൂന്യം) { തിരികെ f_1_ecx[27]; }
    തർക്കം AVX(ശൂന്യം) { തിരികെ f_1_ecx[28]; }
    തർക്കം F16C(ശൂന്യം) { തിരികെ f_1_ecx[29]; }
    തർക്കം RDRAND(ശൂന്യം) { തിരികെ f_1_ecx[30]; }

    തർക്കം MSR(ശൂന്യം) { തിരികെ f_1_edx[5]; }
    തർക്കം CX8(ശൂന്യം) { തിരികെ f_1_edx[8]; }
    തർക്കം SEP(ശൂന്യം) { തിരികെ f_1_edx[11]; }
    തർക്കം CMOV(ശൂന്യം) { തിരികെ f_1_edx[15]; }
    തർക്കം CLFSH(ശൂന്യം) { തിരികെ f_1_edx[19]; }
    തർക്കം MMX(ശൂന്യം) { തിരികെ f_1_edx[23]; }
    തർക്കം FXSR(ശൂന്യം) { തിരികെ f_1_edx[24]; }
    തർക്കം SSE(ശൂന്യം) { തിരികെ f_1_edx[25]; }
    തർക്കം SSE2(ശൂന്യം) { തിരികെ f_1_edx[26]; }

    തർക്കം FSGSBASE(ശൂന്യം) { തിരികെ f_7_ebx[0]; }
    തർക്കം BMI1(ശൂന്യം) { തിരികെ f_7_ebx[3]; }
    തർക്കം HLE(ശൂന്യം) { തിരികെ is_intel && f_7_ebx[4]; }
    തർക്കം AVX2(ശൂന്യം) { തിരികെ f_7_ebx[5]; }
    തർക്കം BMI2(ശൂന്യം) { തിരികെ f_7_ebx[8]; }
    തർക്കം ERMS(ശൂന്യം) { തിരികെ f_7_ebx[9]; }
    തർക്കം INVPCID(ശൂന്യം) { തിരികെ f_7_ebx[10]; }
    തർക്കം RTM(ശൂന്യം) { തിരികെ is_intel && f_7_ebx[11]; }
    തർക്കം AVX512F(ശൂന്യം) { തിരികെ f_7_ebx[16]; }
    തർക്കം AVX512DQ(ശൂന്യം) { തിരികെ f_7_ebx[17]; }
    തർക്കം RDSEED(ശൂന്യം) { തിരികെ f_7_ebx[18]; }
    തർക്കം ADX(ശൂന്യം) { തിരികെ f_7_ebx[19]; }
    തർക്കം AVX512PF(ശൂന്യം) { തിരികെ f_7_ebx[26]; }
    തർക്കം AVX512ER(ശൂന്യം) { തിരികെ f_7_ebx[27]; }
    തർക്കം AVX512CD(ശൂന്യം) { തിരികെ f_7_ebx[28]; }
    തർക്കം AVX512BW(ശൂന്യം) { തിരികെ f_7_ebx[30]; }
    തർക്കം AVX512VL(ശൂന്യം) { തിരികെ f_7_ebx[31]; }

    തർക്കം SHA(ശൂന്യം) { തിരികെ f_7_ebx[29]; }

    തർക്കം PREFETCHWT1(ശൂന്യം) { തിരികെ f_7_ecx[0]; }

    തർക്കം LAHF(ശൂന്യം) { തിരികെ f_81_ecx[0]; }
    തർക്കം LZCNT(ശൂന്യം) { തിരികെ is_intel && f_81_ecx[5]; }
    തർക്കം ABM(ശൂന്യം) { തിരികെ is_amd && f_81_ecx[5]; }
    തർക്കം SSE4a(ശൂന്യം) { തിരികെ is_amd && f_81_ecx[6]; }
    തർക്കം XOP(ശൂന്യം) { തിരികെ is_amd && f_81_ecx[11]; }
    തർക്കം TBM(ശൂന്യം) { തിരികെ is_amd && f_81_ecx[21]; }

    തർക്കം SYSCALL(ശൂന്യം) { തിരികെ is_intel && f_81_edx[11]; }
    തർക്കം MMXEXT(ശൂന്യം) { തിരികെ is_amd && f_81_edx[22]; }
    തർക്കം RDTSCP(ശൂന്യം) { തിരികെ is_intel && f_81_edx[27]; }
    തർക്കം _3DNOWEXT(ശൂന്യം) { തിരികെ is_amd && f_81_edx[30]; }
    തർക്കം _3DNOW(ശൂന്യം) { തിരികെ is_amd && f_81_edx[31]; }

    തർക്കം AVX512_VBMI(ശൂന്യം) { തിരികെ f_7_ecx[1]; }
    തർക്കം AVX512_VNNI(ശൂന്യം) { തിരികെ f_7_ecx[11]; }
    തർക്കം AVX512_FP16(ശൂന്യം) { തിരികെ f_7_edx[23]; }
    തർക്കം AVX512_BF16(ശൂന്യം) { തിരികെ f_7_1_eax[5]; }
    തർക്കം AVX_VNNI(ശൂന്യം) { തിരികെ f_7_1_eax[4]; }

    തർക്കം AMX_TILE(ശൂന്യം) { തിരികെ f_7_edx[24]; }
    തർക്കം AMX_INT8(ശൂന്യം) { തിരികെ f_7_edx[25]; }
    തർക്കം AMX_FP16(ശൂന്യം) { തിരികെ f_7_1_eax[21]; }
    തർക്കം AMX_BF16(ശൂന്യം) { തിരികെ f_7_edx[22]; }

#ifdef _MSC_VER
    സ്ഥിതം ശൂന്യം cpuid(പൂർണ്ണസംഖ്യ cpu_info[4], പൂർണ്ണസംഖ്യ eax) {
        __cpuid(cpu_info, eax);
    }
    സ്ഥിതം ശൂന്യം cpuidex(പൂർണ്ണസംഖ്യ cpu_info[4], പൂർണ്ണസംഖ്യ eax, പൂർണ്ണസംഖ്യ ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#അല്ലെങ്കിൽ
    സ്ഥിതം ശൂന്യം cpuid(പൂർണ്ണസംഖ്യ cpu_info[4], പൂർണ്ണസംഖ്യ eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    സ്ഥിതം ശൂന്യം cpuidex(പൂർണ്ണസംഖ്യ cpu_info[4], പൂർണ്ണസംഖ്യ eax, പൂർണ്ണസംഖ്യ ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<പൂർണ്ണസംഖ്യ, 4> cpui;
        std::vector<std::array<പൂർണ്ണസംഖ്യ, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        പൂർണ്ണസംഖ്യ n_ids = cpui[0];

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        അക്ഷരം vendor[0x20] = {};
        *reinterpret_cast<പൂർണ്ണസംഖ്യ *>(vendor)     = data[0][1];
        *reinterpret_cast<പൂർണ്ണസംഖ്യ *>(vendor + 4) = data[0][3];
        *reinterpret_cast<പൂർണ്ണസംഖ്യ *>(vendor + 8) = data[0][2];
        ഇത്->vendor = vendor;
        എങ്കിൽ (ഇത്->vendor == "GenuineIntel") {
            is_intel = സത്യം;
        } അല്ലെങ്കിൽ എങ്കിൽ (ഇത്->vendor == "AuthenticAMD") {
            is_amd = സത്യം;
        }

        // load bitset with flags for function 0x00000001
        എങ്കിൽ (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        എങ്കിൽ (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned പൂർണ്ണസംഖ്യ n_ex_ids = cpui[0];

        std::vector<std::array<പൂർണ്ണസംഖ്യ, 4>> ext_data;
        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        എങ്കിൽ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        അക്ഷരം brand[0x40] = {};
        എങ്കിൽ (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), വലുപ്പം(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), വലുപ്പം(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), വലുപ്പം(cpui));
            ഇത്->brand = brand;
        }
    }

    തർക്കം is_intel = അസത്യം;
    തർക്കം is_amd = അസത്യം;
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

#എങ്കിൽ 0
ശൂന്യം test_x86_is() {
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

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    പൂർണ്ണസംഖ്യ score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    എങ്കിൽ (!is.FMA()) { തിരികെ 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    എങ്കിൽ (!is.F16C()) { തിരികെ 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    എങ്കിൽ (!is.SSE42()) { തിരികെ 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    എങ്കിൽ (!is.BMI2()) { തിരികെ 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    എങ്കിൽ (!is.AVX()) { തിരികെ 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    എങ്കിൽ (!is.AVX2()) { തിരികെ 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    എങ്കിൽ (!is.AVX_VNNI()) { തിരികെ 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    എങ്കിൽ (!is.AVX512F()) { തിരികെ 0; }
    എങ്കിൽ (!is.AVX512CD()) { തിരികെ 0; }
    എങ്കിൽ (!is.AVX512VL()) { തിരികെ 0; }
    എങ്കിൽ (!is.AVX512DQ()) { തിരികെ 0; }
    എങ്കിൽ (!is.AVX512BW()) { തിരികെ 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    എങ്കിൽ (!is.AVX512_VBMI()) { തിരികെ 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    എങ്കിൽ (!is.AVX512_BF16()) { തിരികെ 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    എങ്കിൽ (!is.AVX512_VNNI()) { തിരികെ 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    എങ്കിൽ (!is.AMX_INT8()) { തിരികെ 0; }
    score += 1<<11;
#endif

    തിരികെ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
