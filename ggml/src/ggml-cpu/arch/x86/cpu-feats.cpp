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
সংরচনা cpuid_x86 {
    তার্কিক SSE3(শূন্য) { প্রত্যাগম f_1_ecx[0]; }
    তার্কিক PCLMULQDQ(শূন্য) { প্রত্যাগম f_1_ecx[1]; }
    তার্কিক MONITOR(শূন্য) { প্রত্যাগম f_1_ecx[3]; }
    তার্কিক SSSE3(শূন্য) { প্রত্যাগম f_1_ecx[9]; }
    তার্কিক FMA(শূন্য) { প্রত্যাগম f_1_ecx[12]; }
    তার্কিক CMPXCHG16B(শূন্য) { প্রত্যাগম f_1_ecx[13]; }
    তার্কিক SSE41(শূন্য) { প্রত্যাগম f_1_ecx[19]; }
    তার্কিক SSE42(শূন্য) { প্রত্যাগম f_1_ecx[20]; }
    তার্কিক MOVBE(শূন্য) { প্রত্যাগম f_1_ecx[22]; }
    তার্কিক POPCNT(শূন্য) { প্রত্যাগম f_1_ecx[23]; }
    তার্কিক AES(শূন্য) { প্রত্যাগম f_1_ecx[25]; }
    তার্কিক XSAVE(শূন্য) { প্রত্যাগম f_1_ecx[26]; }
    তার্কিক OSXSAVE(শূন্য) { প্রত্যাগম f_1_ecx[27]; }
    তার্কিক AVX(শূন্য) { প্রত্যাগম f_1_ecx[28]; }
    তার্কিক F16C(শূন্য) { প্রত্যাগম f_1_ecx[29]; }
    তার্কিক RDRAND(শূন্য) { প্রত্যাগম f_1_ecx[30]; }

    তার্কিক MSR(শূন্য) { প্রত্যাগম f_1_edx[5]; }
    তার্কিক CX8(শূন্য) { প্রত্যাগম f_1_edx[8]; }
    তার্কিক SEP(শূন্য) { প্রত্যাগম f_1_edx[11]; }
    তার্কিক CMOV(শূন্য) { প্রত্যাগম f_1_edx[15]; }
    তার্কিক CLFSH(শূন্য) { প্রত্যাগম f_1_edx[19]; }
    তার্কিক MMX(শূন্য) { প্রত্যাগম f_1_edx[23]; }
    তার্কিক FXSR(শূন্য) { প্রত্যাগম f_1_edx[24]; }
    তার্কিক SSE(শূন্য) { প্রত্যাগম f_1_edx[25]; }
    তার্কিক SSE2(শূন্য) { প্রত্যাগম f_1_edx[26]; }

    তার্কিক FSGSBASE(শূন্য) { প্রত্যাগম f_7_ebx[0]; }
    তার্কিক BMI1(শূন্য) { প্রত্যাগম f_7_ebx[3]; }
    তার্কিক HLE(শূন্য) { প্রত্যাগম is_intel && f_7_ebx[4]; }
    তার্কিক AVX2(শূন্য) { প্রত্যাগম f_7_ebx[5]; }
    তার্কিক BMI2(শূন্য) { প্রত্যাগম f_7_ebx[8]; }
    তার্কিক ERMS(শূন্য) { প্রত্যাগম f_7_ebx[9]; }
    তার্কিক INVPCID(শূন্য) { প্রত্যাগম f_7_ebx[10]; }
    তার্কিক RTM(শূন্য) { প্রত্যাগম is_intel && f_7_ebx[11]; }
    তার্কিক AVX512F(শূন্য) { প্রত্যাগম f_7_ebx[16]; }
    তার্কিক AVX512DQ(শূন্য) { প্রত্যাগম f_7_ebx[17]; }
    তার্কিক RDSEED(শূন্য) { প্রত্যাগম f_7_ebx[18]; }
    তার্কিক ADX(শূন্য) { প্রত্যাগম f_7_ebx[19]; }
    তার্কিক AVX512PF(শূন্য) { প্রত্যাগম f_7_ebx[26]; }
    তার্কিক AVX512ER(শূন্য) { প্রত্যাগম f_7_ebx[27]; }
    তার্কিক AVX512CD(শূন্য) { প্রত্যাগম f_7_ebx[28]; }
    তার্কিক AVX512BW(শূন্য) { প্রত্যাগম f_7_ebx[30]; }
    তার্কিক AVX512VL(শূন্য) { প্রত্যাগম f_7_ebx[31]; }

    তার্কিক SHA(শূন্য) { প্রত্যাগম f_7_ebx[29]; }

    তার্কিক PREFETCHWT1(শূন্য) { প্রত্যাগম f_7_ecx[0]; }

    তার্কিক LAHF(শূন্য) { প্রত্যাগম f_81_ecx[0]; }
    তার্কিক LZCNT(শূন্য) { প্রত্যাগম is_intel && f_81_ecx[5]; }
    তার্কিক ABM(শূন্য) { প্রত্যাগম is_amd && f_81_ecx[5]; }
    তার্কিক SSE4a(শূন্য) { প্রত্যাগম is_amd && f_81_ecx[6]; }
    তার্কিক XOP(শূন্য) { প্রত্যাগম is_amd && f_81_ecx[11]; }
    তার্কিক TBM(শূন্য) { প্রত্যাগম is_amd && f_81_ecx[21]; }

    তার্কিক SYSCALL(শূন্য) { প্রত্যাগম is_intel && f_81_edx[11]; }
    তার্কিক MMXEXT(শূন্য) { প্রত্যাগম is_amd && f_81_edx[22]; }
    তার্কিক RDTSCP(শূন্য) { প্রত্যাগম is_intel && f_81_edx[27]; }
    তার্কিক _3DNOWEXT(শূন্য) { প্রত্যাগম is_amd && f_81_edx[30]; }
    তার্কিক _3DNOW(শূন্য) { প্রত্যাগম is_amd && f_81_edx[31]; }

    তার্কিক AVX512_VBMI(শূন্য) { প্রত্যাগম f_7_ecx[1]; }
    তার্কিক AVX512_VNNI(শূন্য) { প্রত্যাগম f_7_ecx[11]; }
    তার্কিক AVX512_FP16(শূন্য) { প্রত্যাগম f_7_edx[23]; }
    তার্কিক AVX512_BF16(শূন্য) { প্রত্যাগম f_7_1_eax[5]; }
    তার্কিক AVX_VNNI(শূন্য) { প্রত্যাগম f_7_1_eax[4]; }

    তার্কিক AMX_TILE(শূন্য) { প্রত্যাগম f_7_edx[24]; }
    তার্কিক AMX_INT8(শূন্য) { প্রত্যাগম f_7_edx[25]; }
    তার্কিক AMX_FP16(শূন্য) { প্রত্যাগম f_7_1_eax[21]; }
    তার্কিক AMX_BF16(শূন্য) { প্রত্যাগম f_7_edx[22]; }

#ifdef _MSC_VER
    স্থির শূন্য cpuid(পূর্ণাঙ্ক cpu_info[4], পূর্ণাঙ্ক eax) {
        __cpuid(cpu_info, eax);
    }
    স্থির শূন্য cpuidex(পূর্ণাঙ্ক cpu_info[4], পূর্ণাঙ্ক eax, পূর্ণাঙ্ক ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#অন্যথা
    স্থির শূন্য cpuid(পূর্ণাঙ্ক cpu_info[4], পূর্ণাঙ্ক eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    স্থির শূন্য cpuidex(পূর্ণাঙ্ক cpu_info[4], পূর্ণাঙ্ক eax, পূর্ণাঙ্ক ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<পূর্ণাঙ্ক, 4> cpui;
        std::vector<std::array<পূর্ণাঙ্ক, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        পূর্ণাঙ্ক n_ids = cpui[0];

        প্রতি (পূর্ণাঙ্ক i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        বর্ণ vendor[0x20] = {};
        *reinterpret_cast<পূর্ণাঙ্ক *>(vendor)     = data[0][1];
        *reinterpret_cast<পূর্ণাঙ্ক *>(vendor + 4) = data[0][3];
        *reinterpret_cast<পূর্ণাঙ্ক *>(vendor + 8) = data[0][2];
        এতত্->vendor = vendor;
        যদি (এতত্->vendor == "GenuineIntel") {
            is_intel = সত্য;
        } অন্যথা যদি (এতত্->vendor == "AuthenticAMD") {
            is_amd = সত্য;
        }

        // load bitset with flags for function 0x00000001
        যদি (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        যদি (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned পূর্ণাঙ্ক n_ex_ids = cpui[0];

        std::vector<std::array<পূর্ণাঙ্ক, 4>> ext_data;
        প্রতি (unsigned পূর্ণাঙ্ক i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        যদি (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        বর্ণ brand[0x40] = {};
        যদি (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), পরিমাণ(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), পরিমাণ(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), পরিমাণ(cpui));
            এতত্->brand = brand;
        }
    }

    তার্কিক is_intel = অসত্য;
    তার্কিক is_amd = অসত্য;
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

স্থির পূর্ণাঙ্ক ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    পূর্ণাঙ্ক score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    যদি (!is.FMA()) { প্রত্যাগম 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    যদি (!is.F16C()) { প্রত্যাগম 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    যদি (!is.SSE42()) { প্রত্যাগম 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    যদি (!is.BMI2()) { প্রত্যাগম 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    যদি (!is.AVX()) { প্রত্যাগম 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    যদি (!is.AVX2()) { প্রত্যাগম 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    যদি (!is.AVX_VNNI()) { প্রত্যাগম 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    যদি (!is.AVX512F()) { প্রত্যাগম 0; }
    যদি (!is.AVX512CD()) { প্রত্যাগম 0; }
    যদি (!is.AVX512VL()) { প্রত্যাগম 0; }
    যদি (!is.AVX512DQ()) { প্রত্যাগম 0; }
    যদি (!is.AVX512BW()) { প্রত্যাগম 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    যদি (!is.AVX512_VBMI()) { প্রত্যাগম 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    যদি (!is.AVX512_BF16()) { প্রত্যাগম 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    যদি (!is.AVX512_VNNI()) { প্রত্যাগম 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    যদি (!is.AMX_INT8()) { প্রত্যাগম 0; }
    score += 1<<11;
#endif

    প্রত্যাগম score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
