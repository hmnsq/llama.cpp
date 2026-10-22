#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଯଦି defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#ଅନ୍ତର୍ଭାଵ <intrin.h>
#endif

#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <bitset>
#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ସଂରଚନା cpuid_x86 {
    ତାର୍କିକ SSE3(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[0]; }
    ତାର୍କିକ PCLMULQDQ(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[1]; }
    ତାର୍କିକ MONITOR(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[3]; }
    ତାର୍କିକ SSSE3(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[9]; }
    ତାର୍କିକ FMA(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[12]; }
    ତାର୍କିକ CMPXCHG16B(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[13]; }
    ତାର୍କିକ SSE41(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[19]; }
    ତାର୍କିକ SSE42(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[20]; }
    ତାର୍କିକ MOVBE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[22]; }
    ତାର୍କିକ POPCNT(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[23]; }
    ତାର୍କିକ AES(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[25]; }
    ତାର୍କିକ XSAVE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[26]; }
    ତାର୍କିକ OSXSAVE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[27]; }
    ତାର୍କିକ AVX(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[28]; }
    ତାର୍କିକ F16C(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[29]; }
    ତାର୍କିକ RDRAND(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_ecx[30]; }

    ତାର୍କିକ MSR(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[5]; }
    ତାର୍କିକ CX8(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[8]; }
    ତାର୍କିକ SEP(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[11]; }
    ତାର୍କିକ CMOV(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[15]; }
    ତାର୍କିକ CLFSH(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[19]; }
    ତାର୍କିକ MMX(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[23]; }
    ତାର୍କିକ FXSR(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[24]; }
    ତାର୍କିକ SSE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[25]; }
    ତାର୍କିକ SSE2(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_1_edx[26]; }

    ତାର୍କିକ FSGSBASE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[0]; }
    ତାର୍କିକ BMI1(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[3]; }
    ତାର୍କିକ HLE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_intel && f_7_ebx[4]; }
    ତାର୍କିକ AVX2(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[5]; }
    ତାର୍କିକ BMI2(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[8]; }
    ତାର୍କିକ ERMS(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[9]; }
    ତାର୍କିକ INVPCID(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[10]; }
    ତାର୍କିକ RTM(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_intel && f_7_ebx[11]; }
    ତାର୍କିକ AVX512F(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[16]; }
    ତାର୍କିକ AVX512DQ(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[17]; }
    ତାର୍କିକ RDSEED(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[18]; }
    ତାର୍କିକ ADX(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[19]; }
    ତାର୍କିକ AVX512PF(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[26]; }
    ତାର୍କିକ AVX512ER(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[27]; }
    ତାର୍କିକ AVX512CD(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[28]; }
    ତାର୍କିକ AVX512BW(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[30]; }
    ତାର୍କିକ AVX512VL(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[31]; }

    ତାର୍କିକ SHA(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ebx[29]; }

    ତାର୍କିକ PREFETCHWT1(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ecx[0]; }

    ତାର୍କିକ LAHF(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_81_ecx[0]; }
    ତାର୍କିକ LZCNT(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_intel && f_81_ecx[5]; }
    ତାର୍କିକ ABM(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_ecx[5]; }
    ତାର୍କିକ SSE4a(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_ecx[6]; }
    ତାର୍କିକ XOP(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_ecx[11]; }
    ତାର୍କିକ TBM(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_ecx[21]; }

    ତାର୍କିକ SYSCALL(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_intel && f_81_edx[11]; }
    ତାର୍କିକ MMXEXT(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_edx[22]; }
    ତାର୍କିକ RDTSCP(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_intel && f_81_edx[27]; }
    ତାର୍କିକ _3DNOWEXT(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_edx[30]; }
    ତାର୍କିକ _3DNOW(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ is_amd && f_81_edx[31]; }

    ତାର୍କିକ AVX512_VBMI(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ecx[1]; }
    ତାର୍କିକ AVX512_VNNI(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_ecx[11]; }
    ତାର୍କିକ AVX512_FP16(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_edx[23]; }
    ତାର୍କିକ AVX512_BF16(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_1_eax[5]; }
    ତାର୍କିକ AVX_VNNI(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_1_eax[4]; }

    ତାର୍କିକ AMX_TILE(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_edx[24]; }
    ତାର୍କିକ AMX_INT8(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_edx[25]; }
    ତାର୍କିକ AMX_FP16(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_1_eax[21]; }
    ତାର୍କିକ AMX_BF16(ଶୂନ୍ୟ) { ଫେରନ୍ତୁ f_7_edx[22]; }

#ifdef _MSC_VER
    ସ୍ଥିର ଶୂନ୍ୟ cpuid(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_info[4], ପୂର୍ଣ୍ଣସଂଖ୍ୟା eax) {
        __cpuid(cpu_info, eax);
    }
    ସ୍ଥିର ଶୂନ୍ୟ cpuidex(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_info[4], ପୂର୍ଣ୍ଣସଂଖ୍ୟା eax, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ଅନ୍ୟଥା
    ସ୍ଥିର ଶୂନ୍ୟ cpuid(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_info[4], ପୂର୍ଣ୍ଣସଂଖ୍ୟା eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ସ୍ଥିର ଶୂନ୍ୟ cpuidex(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_info[4], ପୂର୍ଣ୍ଣସଂଖ୍ୟା eax, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 4> cpui;
        std::vector<std::array<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = cpui[0];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        ଅକ୍ଷର vendor[0x20] = {};
        *reinterpret_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>(vendor)     = data[0][1];
        *reinterpret_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>(vendor + 4) = data[0][3];
        *reinterpret_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>(vendor + 8) = data[0][2];
        ଏହି->vendor = vendor;
        ଯଦି (ଏହି->vendor == "GenuineIntel") {
            is_intel = ସତ୍ୟ;
        } ଅନ୍ୟଥା ଯଦି (ଏହି->vendor == "AuthenticAMD") {
            is_amd = ସତ୍ୟ;
        }

        // load bitset with flags for function 0x00000001
        ଯଦି (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        ଯଦି (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ex_ids = cpui[0];

        std::vector<std::array<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 4>> ext_data;
        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ଯଦି (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        ଅକ୍ଷର brand[0x40] = {};
        ଯଦି (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), ଆକାର(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), ଆକାର(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), ଆକାର(cpui));
            ଏହି->brand = brand;
        }
    }

    ତାର୍କିକ is_intel = ମିଥ୍ୟା;
    ତାର୍କିକ is_amd = ମିଥ୍ୟା;
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

#ଯଦି 0
ଶୂନ୍ୟ test_x86_is() {
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

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    ଯଦି (!is.FMA()) { ଫେରନ୍ତୁ 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    ଯଦି (!is.F16C()) { ଫେରନ୍ତୁ 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    ଯଦି (!is.SSE42()) { ଫେରନ୍ତୁ 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    ଯଦି (!is.BMI2()) { ଫେରନ୍ତୁ 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    ଯଦି (!is.AVX()) { ଫେରନ୍ତୁ 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    ଯଦି (!is.AVX2()) { ଫେରନ୍ତୁ 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    ଯଦି (!is.AVX_VNNI()) { ଫେରନ୍ତୁ 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    ଯଦି (!is.AVX512F()) { ଫେରନ୍ତୁ 0; }
    ଯଦି (!is.AVX512CD()) { ଫେରନ୍ତୁ 0; }
    ଯଦି (!is.AVX512VL()) { ଫେରନ୍ତୁ 0; }
    ଯଦି (!is.AVX512DQ()) { ଫେରନ୍ତୁ 0; }
    ଯଦି (!is.AVX512BW()) { ଫେରନ୍ତୁ 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    ଯଦି (!is.AVX512_VBMI()) { ଫେରନ୍ତୁ 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    ଯଦି (!is.AVX512_BF16()) { ଫେରନ୍ତୁ 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    ଯଦି (!is.AVX512_VNNI()) { ଫେରନ୍ତୁ 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    ଯଦି (!is.AMX_INT8()) { ଫେରନ୍ତୁ 0; }
    score += 1<<11;
#endif

    ଫେରନ୍ତୁ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
