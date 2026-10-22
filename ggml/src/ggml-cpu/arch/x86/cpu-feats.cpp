#अन्तर्भाव "ggml-backend-impl.h"

#यदि defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#अन्तर्भाव <intrin.h>
#endif

#अन्तर्भाव <cstring>
#अन्तर्भाव <vector>
#अन्तर्भाव <bitset>
#अन्तर्भाव <array>
#अन्तर्भाव <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
संरचना cpuid_x86 {
    तार्किक SSE3(शून्य) { फर्काऊ f_1_ecx[0]; }
    तार्किक PCLMULQDQ(शून्य) { फर्काऊ f_1_ecx[1]; }
    तार्किक MONITOR(शून्य) { फर्काऊ f_1_ecx[3]; }
    तार्किक SSSE3(शून्य) { फर्काऊ f_1_ecx[9]; }
    तार्किक FMA(शून्य) { फर्काऊ f_1_ecx[12]; }
    तार्किक CMPXCHG16B(शून्य) { फर्काऊ f_1_ecx[13]; }
    तार्किक SSE41(शून्य) { फर्काऊ f_1_ecx[19]; }
    तार्किक SSE42(शून्य) { फर्काऊ f_1_ecx[20]; }
    तार्किक MOVBE(शून्य) { फर्काऊ f_1_ecx[22]; }
    तार्किक POPCNT(शून्य) { फर्काऊ f_1_ecx[23]; }
    तार्किक AES(शून्य) { फर्काऊ f_1_ecx[25]; }
    तार्किक XSAVE(शून्य) { फर्काऊ f_1_ecx[26]; }
    तार्किक OSXSAVE(शून्य) { फर्काऊ f_1_ecx[27]; }
    तार्किक AVX(शून्य) { फर्काऊ f_1_ecx[28]; }
    तार्किक F16C(शून्य) { फर्काऊ f_1_ecx[29]; }
    तार्किक RDRAND(शून्य) { फर्काऊ f_1_ecx[30]; }

    तार्किक MSR(शून्य) { फर्काऊ f_1_edx[5]; }
    तार्किक CX8(शून्य) { फर्काऊ f_1_edx[8]; }
    तार्किक SEP(शून्य) { फर्काऊ f_1_edx[11]; }
    तार्किक CMOV(शून्य) { फर्काऊ f_1_edx[15]; }
    तार्किक CLFSH(शून्य) { फर्काऊ f_1_edx[19]; }
    तार्किक MMX(शून्य) { फर्काऊ f_1_edx[23]; }
    तार्किक FXSR(शून्य) { फर्काऊ f_1_edx[24]; }
    तार्किक SSE(शून्य) { फर्काऊ f_1_edx[25]; }
    तार्किक SSE2(शून्य) { फर्काऊ f_1_edx[26]; }

    तार्किक FSGSBASE(शून्य) { फर्काऊ f_7_ebx[0]; }
    तार्किक BMI1(शून्य) { फर्काऊ f_7_ebx[3]; }
    तार्किक HLE(शून्य) { फर्काऊ is_intel && f_7_ebx[4]; }
    तार्किक AVX2(शून्य) { फर्काऊ f_7_ebx[5]; }
    तार्किक BMI2(शून्य) { फर्काऊ f_7_ebx[8]; }
    तार्किक ERMS(शून्य) { फर्काऊ f_7_ebx[9]; }
    तार्किक INVPCID(शून्य) { फर्काऊ f_7_ebx[10]; }
    तार्किक RTM(शून्य) { फर्काऊ is_intel && f_7_ebx[11]; }
    तार्किक AVX512F(शून्य) { फर्काऊ f_7_ebx[16]; }
    तार्किक AVX512DQ(शून्य) { फर्काऊ f_7_ebx[17]; }
    तार्किक RDSEED(शून्य) { फर्काऊ f_7_ebx[18]; }
    तार्किक ADX(शून्य) { फर्काऊ f_7_ebx[19]; }
    तार्किक AVX512PF(शून्य) { फर्काऊ f_7_ebx[26]; }
    तार्किक AVX512ER(शून्य) { फर्काऊ f_7_ebx[27]; }
    तार्किक AVX512CD(शून्य) { फर्काऊ f_7_ebx[28]; }
    तार्किक AVX512BW(शून्य) { फर्काऊ f_7_ebx[30]; }
    तार्किक AVX512VL(शून्य) { फर्काऊ f_7_ebx[31]; }

    तार्किक SHA(शून्य) { फर्काऊ f_7_ebx[29]; }

    तार्किक PREFETCHWT1(शून्य) { फर्काऊ f_7_ecx[0]; }

    तार्किक LAHF(शून्य) { फर्काऊ f_81_ecx[0]; }
    तार्किक LZCNT(शून्य) { फर्काऊ is_intel && f_81_ecx[5]; }
    तार्किक ABM(शून्य) { फर्काऊ is_amd && f_81_ecx[5]; }
    तार्किक SSE4a(शून्य) { फर्काऊ is_amd && f_81_ecx[6]; }
    तार्किक XOP(शून्य) { फर्काऊ is_amd && f_81_ecx[11]; }
    तार्किक TBM(शून्य) { फर्काऊ is_amd && f_81_ecx[21]; }

    तार्किक SYSCALL(शून्य) { फर्काऊ is_intel && f_81_edx[11]; }
    तार्किक MMXEXT(शून्य) { फर्काऊ is_amd && f_81_edx[22]; }
    तार्किक RDTSCP(शून्य) { फर्काऊ is_intel && f_81_edx[27]; }
    तार्किक _3DNOWEXT(शून्य) { फर्काऊ is_amd && f_81_edx[30]; }
    तार्किक _3DNOW(शून्य) { फर्काऊ is_amd && f_81_edx[31]; }

    तार्किक AVX512_VBMI(शून्य) { फर्काऊ f_7_ecx[1]; }
    तार्किक AVX512_VNNI(शून्य) { फर्काऊ f_7_ecx[11]; }
    तार्किक AVX512_FP16(शून्य) { फर्काऊ f_7_edx[23]; }
    तार्किक AVX512_BF16(शून्य) { फर्काऊ f_7_1_eax[5]; }
    तार्किक AVX_VNNI(शून्य) { फर्काऊ f_7_1_eax[4]; }

    तार्किक AMX_TILE(शून्य) { फर्काऊ f_7_edx[24]; }
    तार्किक AMX_INT8(शून्य) { फर्काऊ f_7_edx[25]; }
    तार्किक AMX_FP16(शून्य) { फर्काऊ f_7_1_eax[21]; }
    तार्किक AMX_BF16(शून्य) { फर्काऊ f_7_edx[22]; }

#ifdef _MSC_VER
    स्थिर शून्य cpuid(पूर्णाङ्क cpu_info[4], पूर्णाङ्क eax) {
        __cpuid(cpu_info, eax);
    }
    स्थिर शून्य cpuidex(पूर्णाङ्क cpu_info[4], पूर्णाङ्क eax, पूर्णाङ्क ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#अन्यथा
    स्थिर शून्य cpuid(पूर्णाङ्क cpu_info[4], पूर्णाङ्क eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    स्थिर शून्य cpuidex(पूर्णाङ्क cpu_info[4], पूर्णाङ्क eax, पूर्णाङ्क ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<पूर्णाङ्क, 4> cpui;
        std::vector<std::array<पूर्णाङ्क, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        पूर्णाङ्क n_ids = cpui[0];

        लागि (पूर्णाङ्क i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        वर्ण vendor[0x20] = {};
        *reinterpret_cast<पूर्णाङ्क *>(vendor)     = data[0][1];
        *reinterpret_cast<पूर्णाङ्क *>(vendor + 4) = data[0][3];
        *reinterpret_cast<पूर्णाङ्क *>(vendor + 8) = data[0][2];
        एतत्->vendor = vendor;
        यदि (एतत्->vendor == "GenuineIntel") {
            is_intel = सत्य;
        } अन्यथा यदि (एतत्->vendor == "AuthenticAMD") {
            is_amd = सत्य;
        }

        // load bitset with flags for function 0x00000001
        यदि (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        यदि (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        अचिन्हित पूर्णाङ्क n_ex_ids = cpui[0];

        std::vector<std::array<पूर्णाङ्क, 4>> ext_data;
        लागि (अचिन्हित पूर्णाङ्क i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        यदि (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        वर्ण brand[0x40] = {};
        यदि (n_ex_ids >= 0x80000004) {
            std::स_नकल(brand, ext_data[2].data(), परिमाण(cpui));
            std::स_नकल(brand + 16, ext_data[3].data(), परिमाण(cpui));
            std::स_नकल(brand + 32, ext_data[4].data(), परिमाण(cpui));
            एतत्->brand = brand;
        }
    }

    तार्किक is_intel = असत्य;
    तार्किक is_amd = असत्य;
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

#यदि 0
शून्य test_x86_is() {
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
#endif

स्थिर पूर्णाङ्क ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    पूर्णाङ्क score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    यदि (!is.FMA()) { फर्काऊ 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    यदि (!is.F16C()) { फर्काऊ 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    यदि (!is.SSE42()) { फर्काऊ 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    यदि (!is.BMI2()) { फर्काऊ 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    यदि (!is.AVX()) { फर्काऊ 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    यदि (!is.AVX2()) { फर्काऊ 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    यदि (!is.AVX_VNNI()) { फर्काऊ 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    यदि (!is.AVX512F()) { फर्काऊ 0; }
    यदि (!is.AVX512CD()) { फर्काऊ 0; }
    यदि (!is.AVX512VL()) { फर्काऊ 0; }
    यदि (!is.AVX512DQ()) { फर्काऊ 0; }
    यदि (!is.AVX512BW()) { फर्काऊ 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    यदि (!is.AVX512_VBMI()) { फर्काऊ 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    यदि (!is.AVX512_BF16()) { फर्काऊ 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    यदि (!is.AVX512_VNNI()) { फर्काऊ 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    यदि (!is.AMX_INT8()) { फर्काऊ 0; }
    score += 1<<11;
#endif

    फर्काऊ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
