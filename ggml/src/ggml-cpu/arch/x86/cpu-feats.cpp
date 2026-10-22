#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਜੇ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _MSC_VER
#ਅਨ੍ਤਰ੍ਭਾਵ <intrin.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <bitset>
#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ਬਣਤਰ cpuid_x86 {
    ਤਾਰਕਿਕ SSE3(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[0]; }
    ਤਾਰਕਿਕ PCLMULQDQ(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[1]; }
    ਤਾਰਕਿਕ MONITOR(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[3]; }
    ਤਾਰਕਿਕ SSSE3(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[9]; }
    ਤਾਰਕਿਕ FMA(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[12]; }
    ਤਾਰਕਿਕ CMPXCHG16B(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[13]; }
    ਤਾਰਕਿਕ SSE41(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[19]; }
    ਤਾਰਕਿਕ SSE42(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[20]; }
    ਤਾਰਕਿਕ MOVBE(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[22]; }
    ਤਾਰਕਿਕ POPCNT(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[23]; }
    ਤਾਰਕਿਕ AES(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[25]; }
    ਤਾਰਕਿਕ XSAVE(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[26]; }
    ਤਾਰਕਿਕ OSXSAVE(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[27]; }
    ਤਾਰਕਿਕ AVX(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[28]; }
    ਤਾਰਕਿਕ F16C(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[29]; }
    ਤਾਰਕਿਕ RDRAND(ਖਾਲੀ) { ਵਾਪਸ f_1_ecx[30]; }

    ਤਾਰਕਿਕ MSR(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[5]; }
    ਤਾਰਕਿਕ CX8(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[8]; }
    ਤਾਰਕਿਕ SEP(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[11]; }
    ਤਾਰਕਿਕ CMOV(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[15]; }
    ਤਾਰਕਿਕ CLFSH(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[19]; }
    ਤਾਰਕਿਕ MMX(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[23]; }
    ਤਾਰਕਿਕ FXSR(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[24]; }
    ਤਾਰਕਿਕ SSE(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[25]; }
    ਤਾਰਕਿਕ SSE2(ਖਾਲੀ) { ਵਾਪਸ f_1_edx[26]; }

    ਤਾਰਕਿਕ FSGSBASE(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[0]; }
    ਤਾਰਕਿਕ BMI1(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[3]; }
    ਤਾਰਕਿਕ HLE(ਖਾਲੀ) { ਵਾਪਸ is_intel && f_7_ebx[4]; }
    ਤਾਰਕਿਕ AVX2(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[5]; }
    ਤਾਰਕਿਕ BMI2(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[8]; }
    ਤਾਰਕਿਕ ERMS(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[9]; }
    ਤਾਰਕਿਕ INVPCID(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[10]; }
    ਤਾਰਕਿਕ RTM(ਖਾਲੀ) { ਵਾਪਸ is_intel && f_7_ebx[11]; }
    ਤਾਰਕਿਕ AVX512F(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[16]; }
    ਤਾਰਕਿਕ AVX512DQ(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[17]; }
    ਤਾਰਕਿਕ RDSEED(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[18]; }
    ਤਾਰਕਿਕ ADX(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[19]; }
    ਤਾਰਕਿਕ AVX512PF(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[26]; }
    ਤਾਰਕਿਕ AVX512ER(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[27]; }
    ਤਾਰਕਿਕ AVX512CD(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[28]; }
    ਤਾਰਕਿਕ AVX512BW(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[30]; }
    ਤਾਰਕਿਕ AVX512VL(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[31]; }

    ਤਾਰਕਿਕ SHA(ਖਾਲੀ) { ਵਾਪਸ f_7_ebx[29]; }

    ਤਾਰਕਿਕ PREFETCHWT1(ਖਾਲੀ) { ਵਾਪਸ f_7_ecx[0]; }

    ਤਾਰਕਿਕ LAHF(ਖਾਲੀ) { ਵਾਪਸ f_81_ecx[0]; }
    ਤਾਰਕਿਕ LZCNT(ਖਾਲੀ) { ਵਾਪਸ is_intel && f_81_ecx[5]; }
    ਤਾਰਕਿਕ ABM(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_ecx[5]; }
    ਤਾਰਕਿਕ SSE4a(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_ecx[6]; }
    ਤਾਰਕਿਕ XOP(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_ecx[11]; }
    ਤਾਰਕਿਕ TBM(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_ecx[21]; }

    ਤਾਰਕਿਕ SYSCALL(ਖਾਲੀ) { ਵਾਪਸ is_intel && f_81_edx[11]; }
    ਤਾਰਕਿਕ MMXEXT(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_edx[22]; }
    ਤਾਰਕਿਕ RDTSCP(ਖਾਲੀ) { ਵਾਪਸ is_intel && f_81_edx[27]; }
    ਤਾਰਕਿਕ _3DNOWEXT(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_edx[30]; }
    ਤਾਰਕਿਕ _3DNOW(ਖਾਲੀ) { ਵਾਪਸ is_amd && f_81_edx[31]; }

    ਤਾਰਕਿਕ AVX512_VBMI(ਖਾਲੀ) { ਵਾਪਸ f_7_ecx[1]; }
    ਤਾਰਕਿਕ AVX512_VNNI(ਖਾਲੀ) { ਵਾਪਸ f_7_ecx[11]; }
    ਤਾਰਕਿਕ AVX512_FP16(ਖਾਲੀ) { ਵਾਪਸ f_7_edx[23]; }
    ਤਾਰਕਿਕ AVX512_BF16(ਖਾਲੀ) { ਵਾਪਸ f_7_1_eax[5]; }
    ਤਾਰਕਿਕ AVX_VNNI(ਖਾਲੀ) { ਵਾਪਸ f_7_1_eax[4]; }

    ਤਾਰਕਿਕ AMX_TILE(ਖਾਲੀ) { ਵਾਪਸ f_7_edx[24]; }
    ਤਾਰਕਿਕ AMX_INT8(ਖਾਲੀ) { ਵਾਪਸ f_7_edx[25]; }
    ਤਾਰਕਿਕ AMX_FP16(ਖਾਲੀ) { ਵਾਪਸ f_7_1_eax[21]; }
    ਤਾਰਕਿਕ AMX_BF16(ਖਾਲੀ) { ਵਾਪਸ f_7_edx[22]; }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _MSC_VER
    ਸਥਿਰ ਖਾਲੀ cpuid(ਪੂਰਨ cpu_info[4], ਪੂਰਨ eax) {
        __cpuid(cpu_info, eax);
    }
    ਸਥਿਰ ਖਾਲੀ cpuidex(ਪੂਰਨ cpu_info[4], ਪੂਰਨ eax, ਪੂਰਨ ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ਨਹੀਂਤਾਂ
    ਸਥਿਰ ਖਾਲੀ cpuid(ਪੂਰਨ cpu_info[4], ਪੂਰਨ eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ਸਥਿਰ ਖਾਲੀ cpuidex(ਪੂਰਨ cpu_info[4], ਪੂਰਨ eax, ਪੂਰਨ ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#ਜੇ_ਅੰਤ

    cpuid_x86() {
        std::array<ਪੂਰਨ, 4> cpui;
        std::vector<std::array<ਪੂਰਨ, 4>> ਅੰਕੜੇ;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.ਅੰਕੜੇ(), 0);
        ਪੂਰਨ n_ids = cpui[0];

        ਲਈ (ਪੂਰਨ i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.ਅੰਕੜੇ(), i, 0);
            ਅੰਕੜੇ.push_back(cpui);
        }

        // capture vendor string
        ਅੱਖਰ vendor[0x20] = {};
        *reinterpret_cast<ਪੂਰਨ *>(vendor)     = ਅੰਕੜੇ[0][1];
        *reinterpret_cast<ਪੂਰਨ *>(vendor + 4) = ਅੰਕੜੇ[0][3];
        *reinterpret_cast<ਪੂਰਨ *>(vendor + 8) = ਅੰਕੜੇ[0][2];
        ਇਹ->vendor = vendor;
        ਜੇ (ਇਹ->vendor == "GenuineIntel") {
            is_intel = ਸੱਚ;
        } ਨਹੀਂਤਾਂ ਜੇ (ਇਹ->vendor == "AuthenticAMD") {
            is_amd = ਸੱਚ;
        }

        // load bitset with flags for function 0x00000001
        ਜੇ (n_ids >= 1) {
            f_1_ecx = ਅੰਕੜੇ[1][2];
            f_1_edx = ਅੰਕੜੇ[1][3];
        }

        // load bitset with flags for function 0x00000007
        ਜੇ (n_ids >= 7) {
            f_7_ebx = ਅੰਕੜੇ[7][1];
            f_7_ecx = ਅੰਕੜੇ[7][2];
            f_7_edx = ਅੰਕੜੇ[7][3];
            cpuidex(cpui.ਅੰਕੜੇ(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.ਅੰਕੜੇ(), 0x80000000);
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_ex_ids = cpui[0];

        std::vector<std::array<ਪੂਰਨ, 4>> ext_ਅੰਕੜੇ;
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.ਅੰਕੜੇ(), i, 0);
            ext_ਅੰਕੜੇ.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ਜੇ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_ਅੰਕੜੇ[1][2];
            f_81_edx = ext_ਅੰਕੜੇ[1][3];
        }

        // interpret CPU brand string if reported
        ਅੱਖਰ brand[0x40] = {};
        ਜੇ (n_ex_ids >= 0x80000004) {
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(brand, ext_ਅੰਕੜੇ[2].ਅੰਕੜੇ(), ਆਕਾਰ(cpui));
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(brand + 16, ext_ਅੰਕੜੇ[3].ਅੰਕੜੇ(), ਆਕਾਰ(cpui));
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(brand + 32, ext_ਅੰਕੜੇ[4].ਅੰਕੜੇ(), ਆਕਾਰ(cpui));
            ਇਹ->brand = brand;
        }
    }

    ਤਾਰਕਿਕ is_intel = ਝੂਠ;
    ਤਾਰਕਿਕ is_amd = ਝੂਠ;
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

#ਜੇ 0
ਖਾਲੀ test_x86_is() {
    cpuid_x86 is;
    ਲਿਖੋ("CPU Vendor: %s\n", is.vendor.c_str());
    ਲਿਖੋ("Brand: %s\n", is.brand.c_str());
    ਲਿਖੋ("is_intel: %d\n", is.is_intel);
    ਲਿਖੋ("is_amd: %d\n", is.is_amd);
    ਲਿਖੋ("sse3: %d\n", is.SSE3());
    ਲਿਖੋ("pclmulqdq: %d\n", is.PCLMULQDQ());
    ਲਿਖੋ("ssse3: %d\n", is.SSSE3());
    ਲਿਖੋ("fma: %d\n", is.FMA());
    ਲਿਖੋ("cmpxchg16b: %d\n", is.CMPXCHG16B());
    ਲਿਖੋ("sse41: %d\n", is.SSE41());
    ਲਿਖੋ("sse42: %d\n", is.SSE42());
    ਲਿਖੋ("movbe: %d\n", is.MOVBE());
    ਲਿਖੋ("popcnt: %d\n", is.POPCNT());
    ਲਿਖੋ("aes: %d\n", is.AES());
    ਲਿਖੋ("xsave: %d\n", is.XSAVE());
    ਲਿਖੋ("osxsave: %d\n", is.OSXSAVE());
    ਲਿਖੋ("avx: %d\n", is.AVX());
    ਲਿਖੋ("f16c: %d\n", is.F16C());
    ਲਿਖੋ("rdrand: %d\n", is.RDRAND());
    ਲਿਖੋ("msr: %d\n", is.MSR());
    ਲਿਖੋ("cx8: %d\n", is.CX8());
    ਲਿਖੋ("sep: %d\n", is.SEP());
    ਲਿਖੋ("cmov: %d\n", is.CMOV());
    ਲਿਖੋ("clflush: %d\n", is.CLFSH());
    ਲਿਖੋ("mmx: %d\n", is.MMX());
    ਲਿਖੋ("fxsr: %d\n", is.FXSR());
    ਲਿਖੋ("sse: %d\n", is.SSE());
    ਲਿਖੋ("sse2: %d\n", is.SSE2());
    ਲਿਖੋ("fsgsbase: %d\n", is.FSGSBASE());
    ਲਿਖੋ("bmi1: %d\n", is.BMI1());
    ਲਿਖੋ("hle: %d\n", is.HLE());
    ਲਿਖੋ("avx2: %d\n", is.AVX2());
    ਲਿਖੋ("bmi2: %d\n", is.BMI2());
    ਲਿਖੋ("erms: %d\n", is.ERMS());
    ਲਿਖੋ("invpcid: %d\n", is.INVPCID());
    ਲਿਖੋ("rtm: %d\n", is.RTM());
    ਲਿਖੋ("avx512f: %d\n", is.AVX512F());
    ਲਿਖੋ("rdseed: %d\n", is.RDSEED());
    ਲਿਖੋ("adx: %d\n", is.ADX());
    ਲਿਖੋ("avx512pf: %d\n", is.AVX512PF());
    ਲਿਖੋ("avx512er: %d\n", is.AVX512ER());
    ਲਿਖੋ("avx512cd: %d\n", is.AVX512CD());
    ਲਿਖੋ("sha: %d\n", is.SHA());
    ਲਿਖੋ("prefetchwt1: %d\n", is.PREFETCHWT1());
    ਲਿਖੋ("lahf: %d\n", is.LAHF());
    ਲਿਖੋ("lzcnt: %d\n", is.LZCNT());
    ਲਿਖੋ("abm: %d\n", is.ABM());
    ਲਿਖੋ("sse4a: %d\n", is.SSE4a());
    ਲਿਖੋ("xop: %d\n", is.XOP());
    ਲਿਖੋ("tbm: %d\n", is.TBM());
    ਲਿਖੋ("syscall: %d\n", is.SYSCALL());
    ਲਿਖੋ("mmxext: %d\n", is.MMXEXT());
    ਲਿਖੋ("rdtscp: %d\n", is.RDTSCP());
    ਲਿਖੋ("3dnowext: %d\n", is._3DNOWEXT());
    ਲਿਖੋ("3dnow: %d\n", is._3DNOW());
    ਲਿਖੋ("avx512_vbmi: %d\n", is.AVX512_VBMI());
    ਲਿਖੋ("avx512_vnni: %d\n", is.AVX512_VNNI());
    ਲਿਖੋ("avx512_fp16: %d\n", is.AVX512_FP16());
    ਲਿਖੋ("avx512_bf16: %d\n", is.AVX512_BF16());
    ਲਿਖੋ("amx_tile: %d\n", is.AMX_TILE());
    ਲਿਖੋ("amx_int8: %d\n", is.AMX_INT8());
    ਲਿਖੋ("amx_fp16: %d\n", is.AMX_FP16());
    ਲਿਖੋ("amx_bf16: %d\n", is.AMX_BF16());
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਪੂਰਨ ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    ਪੂਰਨ score = 1;
    cpuid_x86 is;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_FMA
    ਜੇ (!is.FMA()) { ਵਾਪਸ 0; }
    score += 1;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_F16C
    ਜੇ (!is.F16C()) { ਵਾਪਸ 0; }
    score += 1<<1;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SSE42
    ਜੇ (!is.SSE42()) { ਵਾਪਸ 0; }
    score += 1<<2;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_BMI2
    ਜੇ (!is.BMI2()) { ਵਾਪਸ 0; }
    score += 1<<3;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX
    ਜੇ (!is.AVX()) { ਵਾਪਸ 0; }
    score += 1<<4;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX2
    ਜੇ (!is.AVX2()) { ਵਾਪਸ 0; }
    score += 1<<5;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX_VNNI
    ਜੇ (!is.AVX_VNNI()) { ਵਾਪਸ 0; }
    score += 1<<6;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX512
    ਜੇ (!is.AVX512F()) { ਵਾਪਸ 0; }
    ਜੇ (!is.AVX512CD()) { ਵਾਪਸ 0; }
    ਜੇ (!is.AVX512VL()) { ਵਾਪਸ 0; }
    ਜੇ (!is.AVX512DQ()) { ਵਾਪਸ 0; }
    ਜੇ (!is.AVX512BW()) { ਵਾਪਸ 0; }
    score += 1<<7;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX512_VBMI
    ਜੇ (!is.AVX512_VBMI()) { ਵਾਪਸ 0; }
    score += 1<<8;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX512_BF16
    ਜੇ (!is.AVX512_BF16()) { ਵਾਪਸ 0; }
    score += 1<<9;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AVX512_VNNI
    ਜੇ (!is.AVX512_VNNI()) { ਵਾਪਸ 0; }
    score += 1<<10;
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_AMX_INT8
    ਜੇ (!is.AMX_INT8()) { ਵਾਪਸ 0; }
    score += 1<<11;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#ਜੇ_ਅੰਤ // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
