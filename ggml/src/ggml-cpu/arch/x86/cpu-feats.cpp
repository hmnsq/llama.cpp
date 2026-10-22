#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಆದರೆ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _MSC_VER
#ಅನ್ತರ್ಭಾವ <intrin.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <bitset>
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ರಚನೆ cpuid_x86 {
    ತಾರ್ಕಿಕ SSE3(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[0]; }
    ತಾರ್ಕಿಕ PCLMULQDQ(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[1]; }
    ತಾರ್ಕಿಕ MONITOR(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[3]; }
    ತಾರ್ಕಿಕ SSSE3(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[9]; }
    ತಾರ್ಕಿಕ FMA(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[12]; }
    ತಾರ್ಕಿಕ CMPXCHG16B(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[13]; }
    ತಾರ್ಕಿಕ SSE41(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[19]; }
    ತಾರ್ಕಿಕ SSE42(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[20]; }
    ತಾರ್ಕಿಕ MOVBE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[22]; }
    ತಾರ್ಕಿಕ POPCNT(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[23]; }
    ತಾರ್ಕಿಕ AES(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[25]; }
    ತಾರ್ಕಿಕ XSAVE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[26]; }
    ತಾರ್ಕಿಕ OSXSAVE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[27]; }
    ತಾರ್ಕಿಕ AVX(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[28]; }
    ತಾರ್ಕಿಕ F16C(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[29]; }
    ತಾರ್ಕಿಕ RDRAND(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_ecx[30]; }

    ತಾರ್ಕಿಕ MSR(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[5]; }
    ತಾರ್ಕಿಕ CX8(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[8]; }
    ತಾರ್ಕಿಕ SEP(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[11]; }
    ತಾರ್ಕಿಕ CMOV(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[15]; }
    ತಾರ್ಕಿಕ CLFSH(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[19]; }
    ತಾರ್ಕಿಕ MMX(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[23]; }
    ತಾರ್ಕಿಕ FXSR(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[24]; }
    ತಾರ್ಕಿಕ SSE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[25]; }
    ತಾರ್ಕಿಕ SSE2(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_1_edx[26]; }

    ತಾರ್ಕಿಕ FSGSBASE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[0]; }
    ತಾರ್ಕಿಕ BMI1(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[3]; }
    ತಾರ್ಕಿಕ HLE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_intel && f_7_ebx[4]; }
    ತಾರ್ಕಿಕ AVX2(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[5]; }
    ತಾರ್ಕಿಕ BMI2(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[8]; }
    ತಾರ್ಕಿಕ ERMS(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[9]; }
    ತಾರ್ಕಿಕ INVPCID(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[10]; }
    ತಾರ್ಕಿಕ RTM(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_intel && f_7_ebx[11]; }
    ತಾರ್ಕಿಕ AVX512F(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[16]; }
    ತಾರ್ಕಿಕ AVX512DQ(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[17]; }
    ತಾರ್ಕಿಕ RDSEED(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[18]; }
    ತಾರ್ಕಿಕ ADX(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[19]; }
    ತಾರ್ಕಿಕ AVX512PF(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[26]; }
    ತಾರ್ಕಿಕ AVX512ER(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[27]; }
    ತಾರ್ಕಿಕ AVX512CD(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[28]; }
    ತಾರ್ಕಿಕ AVX512BW(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[30]; }
    ತಾರ್ಕಿಕ AVX512VL(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[31]; }

    ತಾರ್ಕಿಕ SHA(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ebx[29]; }

    ತಾರ್ಕಿಕ PREFETCHWT1(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ecx[0]; }

    ತಾರ್ಕಿಕ LAHF(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_81_ecx[0]; }
    ತಾರ್ಕಿಕ LZCNT(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_intel && f_81_ecx[5]; }
    ತಾರ್ಕಿಕ ABM(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_ecx[5]; }
    ತಾರ್ಕಿಕ SSE4a(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_ecx[6]; }
    ತಾರ್ಕಿಕ XOP(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_ecx[11]; }
    ತಾರ್ಕಿಕ TBM(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_ecx[21]; }

    ತಾರ್ಕಿಕ SYSCALL(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_intel && f_81_edx[11]; }
    ತಾರ್ಕಿಕ MMXEXT(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_edx[22]; }
    ತಾರ್ಕಿಕ RDTSCP(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_intel && f_81_edx[27]; }
    ತಾರ್ಕಿಕ _3DNOWEXT(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_edx[30]; }
    ತಾರ್ಕಿಕ _3DNOW(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ is_amd && f_81_edx[31]; }

    ತಾರ್ಕಿಕ AVX512_VBMI(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ecx[1]; }
    ತಾರ್ಕಿಕ AVX512_VNNI(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_ecx[11]; }
    ತಾರ್ಕಿಕ AVX512_FP16(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_edx[23]; }
    ತಾರ್ಕಿಕ AVX512_BF16(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_1_eax[5]; }
    ತಾರ್ಕಿಕ AVX_VNNI(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_1_eax[4]; }

    ತಾರ್ಕಿಕ AMX_TILE(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_edx[24]; }
    ತಾರ್ಕಿಕ AMX_INT8(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_edx[25]; }
    ತಾರ್ಕಿಕ AMX_FP16(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_1_eax[21]; }
    ತಾರ್ಕಿಕ AMX_BF16(ಶೂನ್ಯ) { ಹಿಂತಿರುಗಿ f_7_edx[22]; }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _MSC_VER
    ಸ್ಥಿರ ಶೂನ್ಯ cpuid(ಪೂರ್ಣಾಂಕ cpu_info[4], ಪೂರ್ಣಾಂಕ eax) {
        __cpuid(cpu_info, eax);
    }
    ಸ್ಥಿರ ಶೂನ್ಯ cpuidex(ಪೂರ್ಣಾಂಕ cpu_info[4], ಪೂರ್ಣಾಂಕ eax, ಪೂರ್ಣಾಂಕ ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರ ಶೂನ್ಯ cpuid(ಪೂರ್ಣಾಂಕ cpu_info[4], ಪೂರ್ಣಾಂಕ eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ಸ್ಥಿರ ಶೂನ್ಯ cpuidex(ಪೂರ್ಣಾಂಕ cpu_info[4], ಪೂರ್ಣಾಂಕ eax, ಪೂರ್ಣಾಂಕ ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#ಆದರೆ_ಅಂತ್ಯ

    cpuid_x86() {
        std::array<ಪೂರ್ಣಾಂಕ, 4> cpui;
        std::vector<std::array<ಪೂರ್ಣಾಂಕ, 4>> ದತ್ತಾಂಶ;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.ದತ್ತಾಂಶ(), 0);
        ಪೂರ್ಣಾಂಕ n_ids = cpui[0];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.ದತ್ತಾಂಶ(), i, 0);
            ದತ್ತಾಂಶ.push_back(cpui);
        }

        // capture vendor string
        ಅಕ್ಷರ vendor[0x20] = {};
        *reinterpret_cast<ಪೂರ್ಣಾಂಕ *>(vendor)     = ದತ್ತಾಂಶ[0][1];
        *reinterpret_cast<ಪೂರ್ಣಾಂಕ *>(vendor + 4) = ದತ್ತಾಂಶ[0][3];
        *reinterpret_cast<ಪೂರ್ಣಾಂಕ *>(vendor + 8) = ದತ್ತಾಂಶ[0][2];
        ಇದು->vendor = vendor;
        ಆದರೆ (ಇದು->vendor == "GenuineIntel") {
            is_intel = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಇದು->vendor == "AuthenticAMD") {
            is_amd = ನಿಜ;
        }

        // load bitset with flags for function 0x00000001
        ಆದರೆ (n_ids >= 1) {
            f_1_ecx = ದತ್ತಾಂಶ[1][2];
            f_1_edx = ದತ್ತಾಂಶ[1][3];
        }

        // load bitset with flags for function 0x00000007
        ಆದರೆ (n_ids >= 7) {
            f_7_ebx = ದತ್ತಾಂಶ[7][1];
            f_7_ecx = ದತ್ತಾಂಶ[7][2];
            f_7_edx = ದತ್ತಾಂಶ[7][3];
            cpuidex(cpui.ದತ್ತಾಂಶ(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.ದತ್ತಾಂಶ(), 0x80000000);
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n_ex_ids = cpui[0];

        std::vector<std::array<ಪೂರ್ಣಾಂಕ, 4>> ext_ದತ್ತಾಂಶ;
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.ದತ್ತಾಂಶ(), i, 0);
            ext_ದತ್ತಾಂಶ.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ಆದರೆ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_ದತ್ತಾಂಶ[1][2];
            f_81_edx = ext_ದತ್ತಾಂಶ[1][3];
        }

        // interpret CPU brand string if reported
        ಅಕ್ಷರ brand[0x40] = {};
        ಆದರೆ (n_ex_ids >= 0x80000004) {
            std::ಸ್ಮರಣೆ_ನಕಲು(brand, ext_ದತ್ತಾಂಶ[2].ದತ್ತಾಂಶ(), ಗಾತ್ರ(cpui));
            std::ಸ್ಮರಣೆ_ನಕಲು(brand + 16, ext_ದತ್ತಾಂಶ[3].ದತ್ತಾಂಶ(), ಗಾತ್ರ(cpui));
            std::ಸ್ಮರಣೆ_ನಕಲು(brand + 32, ext_ದತ್ತಾಂಶ[4].ದತ್ತಾಂಶ(), ಗಾತ್ರ(cpui));
            ಇದು->brand = brand;
        }
    }

    ತಾರ್ಕಿಕ is_intel = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ is_amd = ಸುಳ್ಳು;
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

#ಆದರೆ 0
ಶೂನ್ಯ test_x86_is() {
    cpuid_x86 is;
    ಬರೆ("CPU Vendor: %s\n", is.vendor.c_str());
    ಬರೆ("Brand: %s\n", is.brand.c_str());
    ಬರೆ("is_intel: %d\n", is.is_intel);
    ಬರೆ("is_amd: %d\n", is.is_amd);
    ಬರೆ("sse3: %d\n", is.SSE3());
    ಬರೆ("pclmulqdq: %d\n", is.PCLMULQDQ());
    ಬರೆ("ssse3: %d\n", is.SSSE3());
    ಬರೆ("fma: %d\n", is.FMA());
    ಬರೆ("cmpxchg16b: %d\n", is.CMPXCHG16B());
    ಬರೆ("sse41: %d\n", is.SSE41());
    ಬರೆ("sse42: %d\n", is.SSE42());
    ಬರೆ("movbe: %d\n", is.MOVBE());
    ಬರೆ("popcnt: %d\n", is.POPCNT());
    ಬರೆ("aes: %d\n", is.AES());
    ಬರೆ("xsave: %d\n", is.XSAVE());
    ಬರೆ("osxsave: %d\n", is.OSXSAVE());
    ಬರೆ("avx: %d\n", is.AVX());
    ಬರೆ("f16c: %d\n", is.F16C());
    ಬರೆ("rdrand: %d\n", is.RDRAND());
    ಬರೆ("msr: %d\n", is.MSR());
    ಬರೆ("cx8: %d\n", is.CX8());
    ಬರೆ("sep: %d\n", is.SEP());
    ಬರೆ("cmov: %d\n", is.CMOV());
    ಬರೆ("clflush: %d\n", is.CLFSH());
    ಬರೆ("mmx: %d\n", is.MMX());
    ಬರೆ("fxsr: %d\n", is.FXSR());
    ಬರೆ("sse: %d\n", is.SSE());
    ಬರೆ("sse2: %d\n", is.SSE2());
    ಬರೆ("fsgsbase: %d\n", is.FSGSBASE());
    ಬರೆ("bmi1: %d\n", is.BMI1());
    ಬರೆ("hle: %d\n", is.HLE());
    ಬರೆ("avx2: %d\n", is.AVX2());
    ಬರೆ("bmi2: %d\n", is.BMI2());
    ಬರೆ("erms: %d\n", is.ERMS());
    ಬರೆ("invpcid: %d\n", is.INVPCID());
    ಬರೆ("rtm: %d\n", is.RTM());
    ಬರೆ("avx512f: %d\n", is.AVX512F());
    ಬರೆ("rdseed: %d\n", is.RDSEED());
    ಬರೆ("adx: %d\n", is.ADX());
    ಬರೆ("avx512pf: %d\n", is.AVX512PF());
    ಬರೆ("avx512er: %d\n", is.AVX512ER());
    ಬರೆ("avx512cd: %d\n", is.AVX512CD());
    ಬರೆ("sha: %d\n", is.SHA());
    ಬರೆ("prefetchwt1: %d\n", is.PREFETCHWT1());
    ಬರೆ("lahf: %d\n", is.LAHF());
    ಬರೆ("lzcnt: %d\n", is.LZCNT());
    ಬರೆ("abm: %d\n", is.ABM());
    ಬರೆ("sse4a: %d\n", is.SSE4a());
    ಬರೆ("xop: %d\n", is.XOP());
    ಬರೆ("tbm: %d\n", is.TBM());
    ಬರೆ("syscall: %d\n", is.SYSCALL());
    ಬರೆ("mmxext: %d\n", is.MMXEXT());
    ಬರೆ("rdtscp: %d\n", is.RDTSCP());
    ಬರೆ("3dnowext: %d\n", is._3DNOWEXT());
    ಬರೆ("3dnow: %d\n", is._3DNOW());
    ಬರೆ("avx512_vbmi: %d\n", is.AVX512_VBMI());
    ಬರೆ("avx512_vnni: %d\n", is.AVX512_VNNI());
    ಬರೆ("avx512_fp16: %d\n", is.AVX512_FP16());
    ಬರೆ("avx512_bf16: %d\n", is.AVX512_BF16());
    ಬರೆ("amx_tile: %d\n", is.AMX_TILE());
    ಬರೆ("amx_int8: %d\n", is.AMX_INT8());
    ಬರೆ("amx_fp16: %d\n", is.AMX_FP16());
    ಬರೆ("amx_bf16: %d\n", is.AMX_BF16());
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    ಪೂರ್ಣಾಂಕ score = 1;
    cpuid_x86 is;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_FMA
    ಆದರೆ (!is.FMA()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_F16C
    ಆದರೆ (!is.F16C()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<1;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SSE42
    ಆದರೆ (!is.SSE42()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<2;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_BMI2
    ಆದರೆ (!is.BMI2()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<3;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX
    ಆದರೆ (!is.AVX()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<4;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX2
    ಆದರೆ (!is.AVX2()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<5;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX_VNNI
    ಆದರೆ (!is.AVX_VNNI()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<6;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX512
    ಆದರೆ (!is.AVX512F()) { ಹಿಂತಿರುಗಿ 0; }
    ಆದರೆ (!is.AVX512CD()) { ಹಿಂತಿರುಗಿ 0; }
    ಆದರೆ (!is.AVX512VL()) { ಹಿಂತಿರುಗಿ 0; }
    ಆದರೆ (!is.AVX512DQ()) { ಹಿಂತಿರುಗಿ 0; }
    ಆದರೆ (!is.AVX512BW()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<7;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX512_VBMI
    ಆದರೆ (!is.AVX512_VBMI()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<8;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX512_BF16
    ಆದರೆ (!is.AVX512_BF16()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<9;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AVX512_VNNI
    ಆದರೆ (!is.AVX512_VNNI()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<10;
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_AMX_INT8
    ಆದರೆ (!is.AMX_INT8()) { ಹಿಂತಿರುಗಿ 0; }
    score += 1<<11;
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#ಆದರೆ_ಅಂತ್ಯ // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
