#समावेश "ggml-backend-impl.h"

#अगर defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#अगर_घोषित _MSC_VER
#समावेश <intrin.h>
#पूर्ण_अगर

#समावेश <cstring>
#समावेश <vector>
#समावेश <bitset>
#समावेश <array>
#समावेश <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
काष्ठा cpuid_x86 {
    bool SSE3(व्योम) { वापस f_1_ecx[0]; }
    bool PCLMULQDQ(व्योम) { वापस f_1_ecx[1]; }
    bool MONITOR(व्योम) { वापस f_1_ecx[3]; }
    bool SSSE3(व्योम) { वापस f_1_ecx[9]; }
    bool FMA(व्योम) { वापस f_1_ecx[12]; }
    bool CMPXCHG16B(व्योम) { वापस f_1_ecx[13]; }
    bool SSE41(व्योम) { वापस f_1_ecx[19]; }
    bool SSE42(व्योम) { वापस f_1_ecx[20]; }
    bool MOVBE(व्योम) { वापस f_1_ecx[22]; }
    bool POPCNT(व्योम) { वापस f_1_ecx[23]; }
    bool AES(व्योम) { वापस f_1_ecx[25]; }
    bool XSAVE(व्योम) { वापस f_1_ecx[26]; }
    bool OSXSAVE(व्योम) { वापस f_1_ecx[27]; }
    bool AVX(व्योम) { वापस f_1_ecx[28]; }
    bool F16C(व्योम) { वापस f_1_ecx[29]; }
    bool RDRAND(व्योम) { वापस f_1_ecx[30]; }

    bool MSR(व्योम) { वापस f_1_edx[5]; }
    bool CX8(व्योम) { वापस f_1_edx[8]; }
    bool SEP(व्योम) { वापस f_1_edx[11]; }
    bool CMOV(व्योम) { वापस f_1_edx[15]; }
    bool CLFSH(व्योम) { वापस f_1_edx[19]; }
    bool MMX(व्योम) { वापस f_1_edx[23]; }
    bool FXSR(व्योम) { वापस f_1_edx[24]; }
    bool SSE(व्योम) { वापस f_1_edx[25]; }
    bool SSE2(व्योम) { वापस f_1_edx[26]; }

    bool FSGSBASE(व्योम) { वापस f_7_ebx[0]; }
    bool BMI1(व्योम) { वापस f_7_ebx[3]; }
    bool HLE(व्योम) { वापस is_intel && f_7_ebx[4]; }
    bool AVX2(व्योम) { वापस f_7_ebx[5]; }
    bool BMI2(व्योम) { वापस f_7_ebx[8]; }
    bool ERMS(व्योम) { वापस f_7_ebx[9]; }
    bool INVPCID(व्योम) { वापस f_7_ebx[10]; }
    bool RTM(व्योम) { वापस is_intel && f_7_ebx[11]; }
    bool AVX512F(व्योम) { वापस f_7_ebx[16]; }
    bool AVX512DQ(व्योम) { वापस f_7_ebx[17]; }
    bool RDSEED(व्योम) { वापस f_7_ebx[18]; }
    bool ADX(व्योम) { वापस f_7_ebx[19]; }
    bool AVX512PF(व्योम) { वापस f_7_ebx[26]; }
    bool AVX512ER(व्योम) { वापस f_7_ebx[27]; }
    bool AVX512CD(व्योम) { वापस f_7_ebx[28]; }
    bool AVX512BW(व्योम) { वापस f_7_ebx[30]; }
    bool AVX512VL(व्योम) { वापस f_7_ebx[31]; }

    bool SHA(व्योम) { वापस f_7_ebx[29]; }

    bool PREFETCHWT1(व्योम) { वापस f_7_ecx[0]; }

    bool LAHF(व्योम) { वापस f_81_ecx[0]; }
    bool LZCNT(व्योम) { वापस is_intel && f_81_ecx[5]; }
    bool ABM(व्योम) { वापस is_amd && f_81_ecx[5]; }
    bool SSE4a(व्योम) { वापस is_amd && f_81_ecx[6]; }
    bool XOP(व्योम) { वापस is_amd && f_81_ecx[11]; }
    bool TBM(व्योम) { वापस is_amd && f_81_ecx[21]; }

    bool SYSCALL(व्योम) { वापस is_intel && f_81_edx[11]; }
    bool MMXEXT(व्योम) { वापस is_amd && f_81_edx[22]; }
    bool RDTSCP(व्योम) { वापस is_intel && f_81_edx[27]; }
    bool _3DNOWEXT(व्योम) { वापस is_amd && f_81_edx[30]; }
    bool _3DNOW(व्योम) { वापस is_amd && f_81_edx[31]; }

    bool AVX512_VBMI(व्योम) { वापस f_7_ecx[1]; }
    bool AVX512_VNNI(व्योम) { वापस f_7_ecx[11]; }
    bool AVX512_FP16(व्योम) { वापस f_7_edx[23]; }
    bool AVX512_BF16(व्योम) { वापस f_7_1_eax[5]; }
    bool AVX_VNNI(व्योम) { वापस f_7_1_eax[4]; }

    bool AMX_TILE(व्योम) { वापस f_7_edx[24]; }
    bool AMX_INT8(व्योम) { वापस f_7_edx[25]; }
    bool AMX_FP16(व्योम) { वापस f_7_1_eax[21]; }
    bool AMX_BF16(व्योम) { वापस f_7_edx[22]; }

#अगर_घोषित _MSC_VER
    अटल व्योम cpuid(पूर्णांक cpu_info[4], पूर्णांक eax) {
        __cpuid(cpu_info, eax);
    }
    अटल व्योम cpuidex(पूर्णांक cpu_info[4], पूर्णांक eax, पूर्णांक ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#अन्यथा
    अटल व्योम cpuid(पूर्णांक cpu_info[4], पूर्णांक eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    अटल व्योम cpuidex(पूर्णांक cpu_info[4], पूर्णांक eax, पूर्णांक ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#पूर्ण_अगर

    cpuid_x86() {
        std::array<पूर्णांक, 4> cpui;
        std::vector<std::array<पूर्णांक, 4>> तथ्य;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.तथ्य(), 0);
        पूर्णांक n_ids = cpui[0];

        क्रम (पूर्णांक i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.तथ्य(), i, 0);
            तथ्य.push_back(cpui);
        }

        // capture vendor string
        अक्षर vendor[0x20] = {};
        *reinterpret_cast<पूर्णांक *>(vendor)     = तथ्य[0][1];
        *reinterpret_cast<पूर्णांक *>(vendor + 4) = तथ्य[0][3];
        *reinterpret_cast<पूर्णांक *>(vendor + 8) = तथ्य[0][2];
        यह->vendor = vendor;
        अगर (यह->vendor == "GenuineIntel") {
            is_intel = true;
        } अन्यथा अगर (यह->vendor == "AuthenticAMD") {
            is_amd = true;
        }

        // load bitset with flags for function 0x00000001
        अगर (n_ids >= 1) {
            f_1_ecx = तथ्य[1][2];
            f_1_edx = तथ्य[1][3];
        }

        // load bitset with flags for function 0x00000007
        अगर (n_ids >= 7) {
            f_7_ebx = तथ्य[7][1];
            f_7_ecx = तथ्य[7][2];
            f_7_edx = तथ्य[7][3];
            cpuidex(cpui.तथ्य(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.तथ्य(), 0x80000000);
        अचिन्हित पूर्णांक n_ex_ids = cpui[0];

        std::vector<std::array<पूर्णांक, 4>> ext_तथ्य;
        क्रम (अचिन्हित पूर्णांक i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.तथ्य(), i, 0);
            ext_तथ्य.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        अगर (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_तथ्य[1][2];
            f_81_edx = ext_तथ्य[1][3];
        }

        // interpret CPU brand string if reported
        अक्षर brand[0x40] = {};
        अगर (n_ex_ids >= 0x80000004) {
            std::स_नकल(brand, ext_तथ्य[2].तथ्य(), माप(cpui));
            std::स_नकल(brand + 16, ext_तथ्य[3].तथ्य(), माप(cpui));
            std::स_नकल(brand + 32, ext_तथ्य[4].तथ्य(), माप(cpui));
            यह->brand = brand;
        }
    }

    bool is_intel = false;
    bool is_amd = false;
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

#अगर 0
व्योम test_x86_is() {
    cpuid_x86 is;
    म_लिखो("CPU Vendor: %s\n", is.vendor.c_str());
    म_लिखो("Brand: %s\n", is.brand.c_str());
    म_लिखो("is_intel: %d\n", is.is_intel);
    म_लिखो("is_amd: %d\n", is.is_amd);
    म_लिखो("sse3: %d\n", is.SSE3());
    म_लिखो("pclmulqdq: %d\n", is.PCLMULQDQ());
    म_लिखो("ssse3: %d\n", is.SSSE3());
    म_लिखो("fma: %d\n", is.FMA());
    म_लिखो("cmpxchg16b: %d\n", is.CMPXCHG16B());
    म_लिखो("sse41: %d\n", is.SSE41());
    म_लिखो("sse42: %d\n", is.SSE42());
    म_लिखो("movbe: %d\n", is.MOVBE());
    म_लिखो("popcnt: %d\n", is.POPCNT());
    म_लिखो("aes: %d\n", is.AES());
    म_लिखो("xsave: %d\n", is.XSAVE());
    म_लिखो("osxsave: %d\n", is.OSXSAVE());
    म_लिखो("avx: %d\n", is.AVX());
    म_लिखो("f16c: %d\n", is.F16C());
    म_लिखो("rdrand: %d\n", is.RDRAND());
    म_लिखो("msr: %d\n", is.MSR());
    म_लिखो("cx8: %d\n", is.CX8());
    म_लिखो("sep: %d\n", is.SEP());
    म_लिखो("cmov: %d\n", is.CMOV());
    म_लिखो("clflush: %d\n", is.CLFSH());
    म_लिखो("mmx: %d\n", is.MMX());
    म_लिखो("fxsr: %d\n", is.FXSR());
    म_लिखो("sse: %d\n", is.SSE());
    म_लिखो("sse2: %d\n", is.SSE2());
    म_लिखो("fsgsbase: %d\n", is.FSGSBASE());
    म_लिखो("bmi1: %d\n", is.BMI1());
    म_लिखो("hle: %d\n", is.HLE());
    म_लिखो("avx2: %d\n", is.AVX2());
    म_लिखो("bmi2: %d\n", is.BMI2());
    म_लिखो("erms: %d\n", is.ERMS());
    म_लिखो("invpcid: %d\n", is.INVPCID());
    म_लिखो("rtm: %d\n", is.RTM());
    म_लिखो("avx512f: %d\n", is.AVX512F());
    म_लिखो("rdseed: %d\n", is.RDSEED());
    म_लिखो("adx: %d\n", is.ADX());
    म_लिखो("avx512pf: %d\n", is.AVX512PF());
    म_लिखो("avx512er: %d\n", is.AVX512ER());
    म_लिखो("avx512cd: %d\n", is.AVX512CD());
    म_लिखो("sha: %d\n", is.SHA());
    म_लिखो("prefetchwt1: %d\n", is.PREFETCHWT1());
    म_लिखो("lahf: %d\n", is.LAHF());
    म_लिखो("lzcnt: %d\n", is.LZCNT());
    म_लिखो("abm: %d\n", is.ABM());
    म_लिखो("sse4a: %d\n", is.SSE4a());
    म_लिखो("xop: %d\n", is.XOP());
    म_लिखो("tbm: %d\n", is.TBM());
    म_लिखो("syscall: %d\n", is.SYSCALL());
    म_लिखो("mmxext: %d\n", is.MMXEXT());
    म_लिखो("rdtscp: %d\n", is.RDTSCP());
    म_लिखो("3dnowext: %d\n", is._3DNOWEXT());
    म_लिखो("3dnow: %d\n", is._3DNOW());
    म_लिखो("avx512_vbmi: %d\n", is.AVX512_VBMI());
    म_लिखो("avx512_vnni: %d\n", is.AVX512_VNNI());
    म_लिखो("avx512_fp16: %d\n", is.AVX512_FP16());
    म_लिखो("avx512_bf16: %d\n", is.AVX512_BF16());
    म_लिखो("amx_tile: %d\n", is.AMX_TILE());
    म_लिखो("amx_int8: %d\n", is.AMX_INT8());
    म_लिखो("amx_fp16: %d\n", is.AMX_FP16());
    म_लिखो("amx_bf16: %d\n", is.AMX_BF16());
}
#पूर्ण_अगर

अटल पूर्णांक ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    पूर्णांक score = 1;
    cpuid_x86 is;

#अगर_घोषित GGML_FMA
    अगर (!is.FMA()) { वापस 0; }
    score += 1;
#पूर्ण_अगर
#अगर_घोषित GGML_F16C
    अगर (!is.F16C()) { वापस 0; }
    score += 1<<1;
#पूर्ण_अगर
#अगर_घोषित GGML_SSE42
    अगर (!is.SSE42()) { वापस 0; }
    score += 1<<2;
#पूर्ण_अगर
#अगर_घोषित GGML_BMI2
    अगर (!is.BMI2()) { वापस 0; }
    score += 1<<3;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX
    अगर (!is.AVX()) { वापस 0; }
    score += 1<<4;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX2
    अगर (!is.AVX2()) { वापस 0; }
    score += 1<<5;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX_VNNI
    अगर (!is.AVX_VNNI()) { वापस 0; }
    score += 1<<6;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX512
    अगर (!is.AVX512F()) { वापस 0; }
    अगर (!is.AVX512CD()) { वापस 0; }
    अगर (!is.AVX512VL()) { वापस 0; }
    अगर (!is.AVX512DQ()) { वापस 0; }
    अगर (!is.AVX512BW()) { वापस 0; }
    score += 1<<7;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX512_VBMI
    अगर (!is.AVX512_VBMI()) { वापस 0; }
    score += 1<<8;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX512_BF16
    अगर (!is.AVX512_BF16()) { वापस 0; }
    score += 1<<9;
#पूर्ण_अगर
#अगर_घोषित GGML_AVX512_VNNI
    अगर (!is.AVX512_VNNI()) { वापस 0; }
    score += 1<<10;
#पूर्ण_अगर
#अगर_घोषित GGML_AMX_INT8
    अगर (!is.AMX_INT8()) { वापस 0; }
    score += 1<<11;
#पूर्ण_अगर

    वापस score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#पूर्ण_अगर // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
