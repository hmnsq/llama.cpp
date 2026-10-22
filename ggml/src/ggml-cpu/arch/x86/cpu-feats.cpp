#અન્તર્ભાવ "ggml-backend-impl.h"

#જો defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#અન્તર્ભાવ <intrin.h>
#endif

#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <bitset>
#અન્તર્ભાવ <array>
#અન્તર્ભાવ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
સંરચના cpuid_x86 {
    તાર્કિક SSE3(શૂન્ય) { પાછા f_1_ecx[0]; }
    તાર્કિક PCLMULQDQ(શૂન્ય) { પાછા f_1_ecx[1]; }
    તાર્કિક MONITOR(શૂન્ય) { પાછા f_1_ecx[3]; }
    તાર્કિક SSSE3(શૂન્ય) { પાછા f_1_ecx[9]; }
    તાર્કિક FMA(શૂન્ય) { પાછા f_1_ecx[12]; }
    તાર્કિક CMPXCHG16B(શૂન્ય) { પાછા f_1_ecx[13]; }
    તાર્કિક SSE41(શૂન્ય) { પાછા f_1_ecx[19]; }
    તાર્કિક SSE42(શૂન્ય) { પાછા f_1_ecx[20]; }
    તાર્કિક MOVBE(શૂન્ય) { પાછા f_1_ecx[22]; }
    તાર્કિક POPCNT(શૂન્ય) { પાછા f_1_ecx[23]; }
    તાર્કિક AES(શૂન્ય) { પાછા f_1_ecx[25]; }
    તાર્કિક XSAVE(શૂન્ય) { પાછા f_1_ecx[26]; }
    તાર્કિક OSXSAVE(શૂન્ય) { પાછા f_1_ecx[27]; }
    તાર્કિક AVX(શૂન્ય) { પાછા f_1_ecx[28]; }
    તાર્કિક F16C(શૂન્ય) { પાછા f_1_ecx[29]; }
    તાર્કિક RDRAND(શૂન્ય) { પાછા f_1_ecx[30]; }

    તાર્કિક MSR(શૂન્ય) { પાછા f_1_edx[5]; }
    તાર્કિક CX8(શૂન્ય) { પાછા f_1_edx[8]; }
    તાર્કિક SEP(શૂન્ય) { પાછા f_1_edx[11]; }
    તાર્કિક CMOV(શૂન્ય) { પાછા f_1_edx[15]; }
    તાર્કિક CLFSH(શૂન્ય) { પાછા f_1_edx[19]; }
    તાર્કિક MMX(શૂન્ય) { પાછા f_1_edx[23]; }
    તાર્કિક FXSR(શૂન્ય) { પાછા f_1_edx[24]; }
    તાર્કિક SSE(શૂન્ય) { પાછા f_1_edx[25]; }
    તાર્કિક SSE2(શૂન્ય) { પાછા f_1_edx[26]; }

    તાર્કિક FSGSBASE(શૂન્ય) { પાછા f_7_ebx[0]; }
    તાર્કિક BMI1(શૂન્ય) { પાછા f_7_ebx[3]; }
    તાર્કિક HLE(શૂન્ય) { પાછા is_intel && f_7_ebx[4]; }
    તાર્કિક AVX2(શૂન્ય) { પાછા f_7_ebx[5]; }
    તાર્કિક BMI2(શૂન્ય) { પાછા f_7_ebx[8]; }
    તાર્કિક ERMS(શૂન્ય) { પાછા f_7_ebx[9]; }
    તાર્કિક INVPCID(શૂન્ય) { પાછા f_7_ebx[10]; }
    તાર્કિક RTM(શૂન્ય) { પાછા is_intel && f_7_ebx[11]; }
    તાર્કિક AVX512F(શૂન્ય) { પાછા f_7_ebx[16]; }
    તાર્કિક AVX512DQ(શૂન્ય) { પાછા f_7_ebx[17]; }
    તાર્કિક RDSEED(શૂન્ય) { પાછા f_7_ebx[18]; }
    તાર્કિક ADX(શૂન્ય) { પાછા f_7_ebx[19]; }
    તાર્કિક AVX512PF(શૂન્ય) { પાછા f_7_ebx[26]; }
    તાર્કિક AVX512ER(શૂન્ય) { પાછા f_7_ebx[27]; }
    તાર્કિક AVX512CD(શૂન્ય) { પાછા f_7_ebx[28]; }
    તાર્કિક AVX512BW(શૂન્ય) { પાછા f_7_ebx[30]; }
    તાર્કિક AVX512VL(શૂન્ય) { પાછા f_7_ebx[31]; }

    તાર્કિક SHA(શૂન્ય) { પાછા f_7_ebx[29]; }

    તાર્કિક PREFETCHWT1(શૂન્ય) { પાછા f_7_ecx[0]; }

    તાર્કિક LAHF(શૂન્ય) { પાછા f_81_ecx[0]; }
    તાર્કિક LZCNT(શૂન્ય) { પાછા is_intel && f_81_ecx[5]; }
    તાર્કિક ABM(શૂન્ય) { પાછા is_amd && f_81_ecx[5]; }
    તાર્કિક SSE4a(શૂન્ય) { પાછા is_amd && f_81_ecx[6]; }
    તાર્કિક XOP(શૂન્ય) { પાછા is_amd && f_81_ecx[11]; }
    તાર્કિક TBM(શૂન્ય) { પાછા is_amd && f_81_ecx[21]; }

    તાર્કિક SYSCALL(શૂન્ય) { પાછા is_intel && f_81_edx[11]; }
    તાર્કિક MMXEXT(શૂન્ય) { પાછા is_amd && f_81_edx[22]; }
    તાર્કિક RDTSCP(શૂન્ય) { પાછા is_intel && f_81_edx[27]; }
    તાર્કિક _3DNOWEXT(શૂન્ય) { પાછા is_amd && f_81_edx[30]; }
    તાર્કિક _3DNOW(શૂન્ય) { પાછા is_amd && f_81_edx[31]; }

    તાર્કિક AVX512_VBMI(શૂન્ય) { પાછા f_7_ecx[1]; }
    તાર્કિક AVX512_VNNI(શૂન્ય) { પાછા f_7_ecx[11]; }
    તાર્કિક AVX512_FP16(શૂન્ય) { પાછા f_7_edx[23]; }
    તાર્કિક AVX512_BF16(શૂન્ય) { પાછા f_7_1_eax[5]; }
    તાર્કિક AVX_VNNI(શૂન્ય) { પાછા f_7_1_eax[4]; }

    તાર્કિક AMX_TILE(શૂન્ય) { પાછા f_7_edx[24]; }
    તાર્કિક AMX_INT8(શૂન્ય) { પાછા f_7_edx[25]; }
    તાર્કિક AMX_FP16(શૂન્ય) { પાછા f_7_1_eax[21]; }
    તાર્કિક AMX_BF16(શૂન્ય) { પાછા f_7_edx[22]; }

#ifdef _MSC_VER
    સ્થિર શૂન્ય cpuid(પૂર્ણાંક cpu_info[4], પૂર્ણાંક eax) {
        __cpuid(cpu_info, eax);
    }
    સ્થિર શૂન્ય cpuidex(પૂર્ણાંક cpu_info[4], પૂર્ણાંક eax, પૂર્ણાંક ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#નહીંતર
    સ્થિર શૂન્ય cpuid(પૂર્ણાંક cpu_info[4], પૂર્ણાંક eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    સ્થિર શૂન્ય cpuidex(પૂર્ણાંક cpu_info[4], પૂર્ણાંક eax, પૂર્ણાંક ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<પૂર્ણાંક, 4> cpui;
        std::vector<std::array<પૂર્ણાંક, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        પૂર્ણાંક n_ids = cpui[0];

        માટે (પૂર્ણાંક i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        અક્ષર vendor[0x20] = {};
        *reinterpret_cast<પૂર્ણાંક *>(vendor)     = data[0][1];
        *reinterpret_cast<પૂર્ણાંક *>(vendor + 4) = data[0][3];
        *reinterpret_cast<પૂર્ણાંક *>(vendor + 8) = data[0][2];
        આ->vendor = vendor;
        જો (આ->vendor == "GenuineIntel") {
            is_intel = સાચું;
        } નહીંતર જો (આ->vendor == "AuthenticAMD") {
            is_amd = સાચું;
        }

        // load bitset with flags for function 0x00000001
        જો (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        જો (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned પૂર્ણાંક n_ex_ids = cpui[0];

        std::vector<std::array<પૂર્ણાંક, 4>> ext_data;
        માટે (unsigned પૂર્ણાંક i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        જો (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        અક્ષર brand[0x40] = {};
        જો (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), કદ(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), કદ(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), કદ(cpui));
            આ->brand = brand;
        }
    }

    તાર્કિક is_intel = ખોટું;
    તાર્કિક is_amd = ખોટું;
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

#જો 0
શૂન્ય test_x86_is() {
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

સ્થિર પૂર્ણાંક ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    પૂર્ણાંક score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    જો (!is.FMA()) { પાછા 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    જો (!is.F16C()) { પાછા 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    જો (!is.SSE42()) { પાછા 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    જો (!is.BMI2()) { પાછા 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    જો (!is.AVX()) { પાછા 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    જો (!is.AVX2()) { પાછા 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    જો (!is.AVX_VNNI()) { પાછા 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    જો (!is.AVX512F()) { પાછા 0; }
    જો (!is.AVX512CD()) { પાછા 0; }
    જો (!is.AVX512VL()) { પાછા 0; }
    જો (!is.AVX512DQ()) { પાછા 0; }
    જો (!is.AVX512BW()) { પાછા 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    જો (!is.AVX512_VBMI()) { પાછા 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    જો (!is.AVX512_BF16()) { પાછા 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    જો (!is.AVX512_VNNI()) { પાછા 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    જો (!is.AMX_INT8()) { પાછા 0; }
    score += 1<<11;
#endif

    પાછા score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
