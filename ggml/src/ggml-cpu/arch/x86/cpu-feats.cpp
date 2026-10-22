#ضمن "ggml-backend-impl.h"

#ܐܢ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#ضمن <intrin.h>
#endif

#ضمن <cstring>
#ضمن <vector>
#ضمن <bitset>
#ضمن <array>
#ضمن <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
بنية cpuid_x86 {
    منطقي SSE3(فراغ) { ܦܢܐ f_1_ecx[0]; }
    منطقي PCLMULQDQ(فراغ) { ܦܢܐ f_1_ecx[1]; }
    منطقي MONITOR(فراغ) { ܦܢܐ f_1_ecx[3]; }
    منطقي SSSE3(فراغ) { ܦܢܐ f_1_ecx[9]; }
    منطقي FMA(فراغ) { ܦܢܐ f_1_ecx[12]; }
    منطقي CMPXCHG16B(فراغ) { ܦܢܐ f_1_ecx[13]; }
    منطقي SSE41(فراغ) { ܦܢܐ f_1_ecx[19]; }
    منطقي SSE42(فراغ) { ܦܢܐ f_1_ecx[20]; }
    منطقي MOVBE(فراغ) { ܦܢܐ f_1_ecx[22]; }
    منطقي POPCNT(فراغ) { ܦܢܐ f_1_ecx[23]; }
    منطقي AES(فراغ) { ܦܢܐ f_1_ecx[25]; }
    منطقي XSAVE(فراغ) { ܦܢܐ f_1_ecx[26]; }
    منطقي OSXSAVE(فراغ) { ܦܢܐ f_1_ecx[27]; }
    منطقي AVX(فراغ) { ܦܢܐ f_1_ecx[28]; }
    منطقي F16C(فراغ) { ܦܢܐ f_1_ecx[29]; }
    منطقي RDRAND(فراغ) { ܦܢܐ f_1_ecx[30]; }

    منطقي MSR(فراغ) { ܦܢܐ f_1_edx[5]; }
    منطقي CX8(فراغ) { ܦܢܐ f_1_edx[8]; }
    منطقي SEP(فراغ) { ܦܢܐ f_1_edx[11]; }
    منطقي CMOV(فراغ) { ܦܢܐ f_1_edx[15]; }
    منطقي CLFSH(فراغ) { ܦܢܐ f_1_edx[19]; }
    منطقي MMX(فراغ) { ܦܢܐ f_1_edx[23]; }
    منطقي FXSR(فراغ) { ܦܢܐ f_1_edx[24]; }
    منطقي SSE(فراغ) { ܦܢܐ f_1_edx[25]; }
    منطقي SSE2(فراغ) { ܦܢܐ f_1_edx[26]; }

    منطقي FSGSBASE(فراغ) { ܦܢܐ f_7_ebx[0]; }
    منطقي BMI1(فراغ) { ܦܢܐ f_7_ebx[3]; }
    منطقي HLE(فراغ) { ܦܢܐ is_intel && f_7_ebx[4]; }
    منطقي AVX2(فراغ) { ܦܢܐ f_7_ebx[5]; }
    منطقي BMI2(فراغ) { ܦܢܐ f_7_ebx[8]; }
    منطقي ERMS(فراغ) { ܦܢܐ f_7_ebx[9]; }
    منطقي INVPCID(فراغ) { ܦܢܐ f_7_ebx[10]; }
    منطقي RTM(فراغ) { ܦܢܐ is_intel && f_7_ebx[11]; }
    منطقي AVX512F(فراغ) { ܦܢܐ f_7_ebx[16]; }
    منطقي AVX512DQ(فراغ) { ܦܢܐ f_7_ebx[17]; }
    منطقي RDSEED(فراغ) { ܦܢܐ f_7_ebx[18]; }
    منطقي ADX(فراغ) { ܦܢܐ f_7_ebx[19]; }
    منطقي AVX512PF(فراغ) { ܦܢܐ f_7_ebx[26]; }
    منطقي AVX512ER(فراغ) { ܦܢܐ f_7_ebx[27]; }
    منطقي AVX512CD(فراغ) { ܦܢܐ f_7_ebx[28]; }
    منطقي AVX512BW(فراغ) { ܦܢܐ f_7_ebx[30]; }
    منطقي AVX512VL(فراغ) { ܦܢܐ f_7_ebx[31]; }

    منطقي SHA(فراغ) { ܦܢܐ f_7_ebx[29]; }

    منطقي PREFETCHWT1(فراغ) { ܦܢܐ f_7_ecx[0]; }

    منطقي LAHF(فراغ) { ܦܢܐ f_81_ecx[0]; }
    منطقي LZCNT(فراغ) { ܦܢܐ is_intel && f_81_ecx[5]; }
    منطقي ABM(فراغ) { ܦܢܐ is_amd && f_81_ecx[5]; }
    منطقي SSE4a(فراغ) { ܦܢܐ is_amd && f_81_ecx[6]; }
    منطقي XOP(فراغ) { ܦܢܐ is_amd && f_81_ecx[11]; }
    منطقي TBM(فراغ) { ܦܢܐ is_amd && f_81_ecx[21]; }

    منطقي SYSCALL(فراغ) { ܦܢܐ is_intel && f_81_edx[11]; }
    منطقي MMXEXT(فراغ) { ܦܢܐ is_amd && f_81_edx[22]; }
    منطقي RDTSCP(فراغ) { ܦܢܐ is_intel && f_81_edx[27]; }
    منطقي _3DNOWEXT(فراغ) { ܦܢܐ is_amd && f_81_edx[30]; }
    منطقي _3DNOW(فراغ) { ܦܢܐ is_amd && f_81_edx[31]; }

    منطقي AVX512_VBMI(فراغ) { ܦܢܐ f_7_ecx[1]; }
    منطقي AVX512_VNNI(فراغ) { ܦܢܐ f_7_ecx[11]; }
    منطقي AVX512_FP16(فراغ) { ܦܢܐ f_7_edx[23]; }
    منطقي AVX512_BF16(فراغ) { ܦܢܐ f_7_1_eax[5]; }
    منطقي AVX_VNNI(فراغ) { ܦܢܐ f_7_1_eax[4]; }

    منطقي AMX_TILE(فراغ) { ܦܢܐ f_7_edx[24]; }
    منطقي AMX_INT8(فراغ) { ܦܢܐ f_7_edx[25]; }
    منطقي AMX_FP16(فراغ) { ܦܢܐ f_7_1_eax[21]; }
    منطقي AMX_BF16(فراغ) { ܦܢܐ f_7_edx[22]; }

#ifdef _MSC_VER
    ساكن فراغ cpuid(صحيح cpu_info[4], صحيح eax) {
        __cpuid(cpu_info, eax);
    }
    ساكن فراغ cpuidex(صحيح cpu_info[4], صحيح eax, صحيح ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ܐܚܪܢܐ
    ساكن فراغ cpuid(صحيح cpu_info[4], صحيح eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ساكن فراغ cpuidex(صحيح cpu_info[4], صحيح eax, صحيح ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<صحيح, 4> cpui;
        std::vector<std::array<صحيح, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        صحيح n_ids = cpui[0];

        لكل (صحيح i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        حرف vendor[0x20] = {};
        *reinterpret_cast<صحيح *>(vendor)     = data[0][1];
        *reinterpret_cast<صحيح *>(vendor + 4) = data[0][3];
        *reinterpret_cast<صحيح *>(vendor + 8) = data[0][2];
        هذا->vendor = vendor;
        ܐܢ (هذا->vendor == "GenuineIntel") {
            is_intel = ܫܪܪܐ;
        } ܐܚܪܢܐ ܐܢ (هذا->vendor == "AuthenticAMD") {
            is_amd = ܫܪܪܐ;
        }

        // load bitset with flags for function 0x00000001
        ܐܢ (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        ܐܢ (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned صحيح n_ex_ids = cpui[0];

        std::vector<std::array<صحيح, 4>> ext_data;
        لكل (unsigned صحيح i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ܐܢ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        حرف brand[0x40] = {};
        ܐܢ (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), حجم(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), حجم(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), حجم(cpui));
            هذا->brand = brand;
        }
    }

    منطقي is_intel = ܕܓܠܐ;
    منطقي is_amd = ܕܓܠܐ;
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

#ܐܢ 0
فراغ test_x86_is() {
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

ساكن صحيح ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    صحيح score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    ܐܢ (!is.FMA()) { ܦܢܐ 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    ܐܢ (!is.F16C()) { ܦܢܐ 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    ܐܢ (!is.SSE42()) { ܦܢܐ 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    ܐܢ (!is.BMI2()) { ܦܢܐ 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    ܐܢ (!is.AVX()) { ܦܢܐ 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    ܐܢ (!is.AVX2()) { ܦܢܐ 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    ܐܢ (!is.AVX_VNNI()) { ܦܢܐ 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    ܐܢ (!is.AVX512F()) { ܦܢܐ 0; }
    ܐܢ (!is.AVX512CD()) { ܦܢܐ 0; }
    ܐܢ (!is.AVX512VL()) { ܦܢܐ 0; }
    ܐܢ (!is.AVX512DQ()) { ܦܢܐ 0; }
    ܐܢ (!is.AVX512BW()) { ܦܢܐ 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    ܐܢ (!is.AVX512_VBMI()) { ܦܢܐ 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    ܐܢ (!is.AVX512_BF16()) { ܦܢܐ 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    ܐܢ (!is.AVX512_VNNI()) { ܦܢܐ 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    ܐܢ (!is.AMX_INT8()) { ܦܢܐ 0; }
    score += 1<<11;
#endif

    ܦܢܐ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
