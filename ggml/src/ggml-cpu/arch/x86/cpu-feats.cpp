#شامل "ggml-backend-impl.h"

#اگر defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#شامل <intrin.h>
#endif

#شامل <cstring>
#شامل <vector>
#شامل <bitset>
#شامل <array>
#شامل <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ساختار cpuid_x86 {
    منطقی SSE3(تهی) { بازگشت f_1_ecx[0]; }
    منطقی PCLMULQDQ(تهی) { بازگشت f_1_ecx[1]; }
    منطقی MONITOR(تهی) { بازگشت f_1_ecx[3]; }
    منطقی SSSE3(تهی) { بازگشت f_1_ecx[9]; }
    منطقی FMA(تهی) { بازگشت f_1_ecx[12]; }
    منطقی CMPXCHG16B(تهی) { بازگشت f_1_ecx[13]; }
    منطقی SSE41(تهی) { بازگشت f_1_ecx[19]; }
    منطقی SSE42(تهی) { بازگشت f_1_ecx[20]; }
    منطقی MOVBE(تهی) { بازگشت f_1_ecx[22]; }
    منطقی POPCNT(تهی) { بازگشت f_1_ecx[23]; }
    منطقی AES(تهی) { بازگشت f_1_ecx[25]; }
    منطقی XSAVE(تهی) { بازگشت f_1_ecx[26]; }
    منطقی OSXSAVE(تهی) { بازگشت f_1_ecx[27]; }
    منطقی AVX(تهی) { بازگشت f_1_ecx[28]; }
    منطقی F16C(تهی) { بازگشت f_1_ecx[29]; }
    منطقی RDRAND(تهی) { بازگشت f_1_ecx[30]; }

    منطقی MSR(تهی) { بازگشت f_1_edx[5]; }
    منطقی CX8(تهی) { بازگشت f_1_edx[8]; }
    منطقی SEP(تهی) { بازگشت f_1_edx[11]; }
    منطقی CMOV(تهی) { بازگشت f_1_edx[15]; }
    منطقی CLFSH(تهی) { بازگشت f_1_edx[19]; }
    منطقی MMX(تهی) { بازگشت f_1_edx[23]; }
    منطقی FXSR(تهی) { بازگشت f_1_edx[24]; }
    منطقی SSE(تهی) { بازگشت f_1_edx[25]; }
    منطقی SSE2(تهی) { بازگشت f_1_edx[26]; }

    منطقی FSGSBASE(تهی) { بازگشت f_7_ebx[0]; }
    منطقی BMI1(تهی) { بازگشت f_7_ebx[3]; }
    منطقی HLE(تهی) { بازگشت is_intel && f_7_ebx[4]; }
    منطقی AVX2(تهی) { بازگشت f_7_ebx[5]; }
    منطقی BMI2(تهی) { بازگشت f_7_ebx[8]; }
    منطقی ERMS(تهی) { بازگشت f_7_ebx[9]; }
    منطقی INVPCID(تهی) { بازگشت f_7_ebx[10]; }
    منطقی RTM(تهی) { بازگشت is_intel && f_7_ebx[11]; }
    منطقی AVX512F(تهی) { بازگشت f_7_ebx[16]; }
    منطقی AVX512DQ(تهی) { بازگشت f_7_ebx[17]; }
    منطقی RDSEED(تهی) { بازگشت f_7_ebx[18]; }
    منطقی ADX(تهی) { بازگشت f_7_ebx[19]; }
    منطقی AVX512PF(تهی) { بازگشت f_7_ebx[26]; }
    منطقی AVX512ER(تهی) { بازگشت f_7_ebx[27]; }
    منطقی AVX512CD(تهی) { بازگشت f_7_ebx[28]; }
    منطقی AVX512BW(تهی) { بازگشت f_7_ebx[30]; }
    منطقی AVX512VL(تهی) { بازگشت f_7_ebx[31]; }

    منطقی SHA(تهی) { بازگشت f_7_ebx[29]; }

    منطقی PREFETCHWT1(تهی) { بازگشت f_7_ecx[0]; }

    منطقی LAHF(تهی) { بازگشت f_81_ecx[0]; }
    منطقی LZCNT(تهی) { بازگشت is_intel && f_81_ecx[5]; }
    منطقی ABM(تهی) { بازگشت is_amd && f_81_ecx[5]; }
    منطقی SSE4a(تهی) { بازگشت is_amd && f_81_ecx[6]; }
    منطقی XOP(تهی) { بازگشت is_amd && f_81_ecx[11]; }
    منطقی TBM(تهی) { بازگشت is_amd && f_81_ecx[21]; }

    منطقی SYSCALL(تهی) { بازگشت is_intel && f_81_edx[11]; }
    منطقی MMXEXT(تهی) { بازگشت is_amd && f_81_edx[22]; }
    منطقی RDTSCP(تهی) { بازگشت is_intel && f_81_edx[27]; }
    منطقی _3DNOWEXT(تهی) { بازگشت is_amd && f_81_edx[30]; }
    منطقی _3DNOW(تهی) { بازگشت is_amd && f_81_edx[31]; }

    منطقی AVX512_VBMI(تهی) { بازگشت f_7_ecx[1]; }
    منطقی AVX512_VNNI(تهی) { بازگشت f_7_ecx[11]; }
    منطقی AVX512_FP16(تهی) { بازگشت f_7_edx[23]; }
    منطقی AVX512_BF16(تهی) { بازگشت f_7_1_eax[5]; }
    منطقی AVX_VNNI(تهی) { بازگشت f_7_1_eax[4]; }

    منطقی AMX_TILE(تهی) { بازگشت f_7_edx[24]; }
    منطقی AMX_INT8(تهی) { بازگشت f_7_edx[25]; }
    منطقی AMX_FP16(تهی) { بازگشت f_7_1_eax[21]; }
    منطقی AMX_BF16(تهی) { بازگشت f_7_edx[22]; }

#ifdef _MSC_VER
    ایستا تهی cpuid(صحیح cpu_info[4], صحیح eax) {
        __cpuid(cpu_info, eax);
    }
    ایستا تهی cpuidex(صحیح cpu_info[4], صحیح eax, صحیح ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#وگرنه
    ایستا تهی cpuid(صحیح cpu_info[4], صحیح eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ایستا تهی cpuidex(صحیح cpu_info[4], صحیح eax, صحیح ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<صحیح, 4> cpui;
        std::vector<std::array<صحیح, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        صحیح n_ids = cpui[0];

        برای (صحیح i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        نویسه vendor[0x20] = {};
        *reinterpret_cast<صحیح *>(vendor)     = data[0][1];
        *reinterpret_cast<صحیح *>(vendor + 4) = data[0][3];
        *reinterpret_cast<صحیح *>(vendor + 8) = data[0][2];
        این->vendor = vendor;
        اگر (این->vendor == "GenuineIntel") {
            is_intel = درست;
        } وگرنه اگر (این->vendor == "AuthenticAMD") {
            is_amd = درست;
        }

        // load bitset with flags for function 0x00000001
        اگر (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        اگر (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned صحیح n_ex_ids = cpui[0];

        std::vector<std::array<صحیح, 4>> ext_data;
        برای (unsigned صحیح i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        اگر (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        نویسه brand[0x40] = {};
        اگر (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), اندازه(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), اندازه(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), اندازه(cpui));
            این->brand = brand;
        }
    }

    منطقی is_intel = نادرست;
    منطقی is_amd = نادرست;
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

#اگر 0
تهی test_x86_is() {
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

ایستا صحیح ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    صحیح score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    اگر (!is.FMA()) { بازگشت 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    اگر (!is.F16C()) { بازگشت 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    اگر (!is.SSE42()) { بازگشت 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    اگر (!is.BMI2()) { بازگشت 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    اگر (!is.AVX()) { بازگشت 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    اگر (!is.AVX2()) { بازگشت 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    اگر (!is.AVX_VNNI()) { بازگشت 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    اگر (!is.AVX512F()) { بازگشت 0; }
    اگر (!is.AVX512CD()) { بازگشت 0; }
    اگر (!is.AVX512VL()) { بازگشت 0; }
    اگر (!is.AVX512DQ()) { بازگشت 0; }
    اگر (!is.AVX512BW()) { بازگشت 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    اگر (!is.AVX512_VBMI()) { بازگشت 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    اگر (!is.AVX512_BF16()) { بازگشت 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    اگر (!is.AVX512_VNNI()) { بازگشت 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    اگر (!is.AMX_INT8()) { بازگشت 0; }
    score += 1<<11;
#endif

    بازگشت score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
