#अन्तर्भाव "ggml-backend-impl.h"

#यदि defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#यदि_परिभाषितम् _MSC_VER
#अन्तर्भाव <intrin.h>
#यदि_अन्तः

#अन्तर्भाव <cstring>
#अन्तर्भाव <vector>
#अन्तर्भाव <bitset>
#अन्तर्भाव <array>
#अन्तर्भाव <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
संरचना cpuid_x86 {
    तार्किक SSE3(शून्य) { प्रत्यागम f_1_ecx[0]; }
    तार्किक PCLMULQDQ(शून्य) { प्रत्यागम f_1_ecx[1]; }
    तार्किक MONITOR(शून्य) { प्रत्यागम f_1_ecx[3]; }
    तार्किक SSSE3(शून्य) { प्रत्यागम f_1_ecx[9]; }
    तार्किक FMA(शून्य) { प्रत्यागम f_1_ecx[12]; }
    तार्किक CMPXCHG16B(शून्य) { प्रत्यागम f_1_ecx[13]; }
    तार्किक SSE41(शून्य) { प्रत्यागम f_1_ecx[19]; }
    तार्किक SSE42(शून्य) { प्रत्यागम f_1_ecx[20]; }
    तार्किक MOVBE(शून्य) { प्रत्यागम f_1_ecx[22]; }
    तार्किक POPCNT(शून्य) { प्रत्यागम f_1_ecx[23]; }
    तार्किक AES(शून्य) { प्रत्यागम f_1_ecx[25]; }
    तार्किक XSAVE(शून्य) { प्रत्यागम f_1_ecx[26]; }
    तार्किक OSXSAVE(शून्य) { प्रत्यागम f_1_ecx[27]; }
    तार्किक AVX(शून्य) { प्रत्यागम f_1_ecx[28]; }
    तार्किक F16C(शून्य) { प्रत्यागम f_1_ecx[29]; }
    तार्किक RDRAND(शून्य) { प्रत्यागम f_1_ecx[30]; }

    तार्किक MSR(शून्य) { प्रत्यागम f_1_edx[5]; }
    तार्किक CX8(शून्य) { प्रत्यागम f_1_edx[8]; }
    तार्किक SEP(शून्य) { प्रत्यागम f_1_edx[11]; }
    तार्किक CMOV(शून्य) { प्रत्यागम f_1_edx[15]; }
    तार्किक CLFSH(शून्य) { प्रत्यागम f_1_edx[19]; }
    तार्किक MMX(शून्य) { प्रत्यागम f_1_edx[23]; }
    तार्किक FXSR(शून्य) { प्रत्यागम f_1_edx[24]; }
    तार्किक SSE(शून्य) { प्रत्यागम f_1_edx[25]; }
    तार्किक SSE2(शून्य) { प्रत्यागम f_1_edx[26]; }

    तार्किक FSGSBASE(शून्य) { प्रत्यागम f_7_ebx[0]; }
    तार्किक BMI1(शून्य) { प्रत्यागम f_7_ebx[3]; }
    तार्किक HLE(शून्य) { प्रत्यागम is_intel && f_7_ebx[4]; }
    तार्किक AVX2(शून्य) { प्रत्यागम f_7_ebx[5]; }
    तार्किक BMI2(शून्य) { प्रत्यागम f_7_ebx[8]; }
    तार्किक ERMS(शून्य) { प्रत्यागम f_7_ebx[9]; }
    तार्किक INVPCID(शून्य) { प्रत्यागम f_7_ebx[10]; }
    तार्किक RTM(शून्य) { प्रत्यागम is_intel && f_7_ebx[11]; }
    तार्किक AVX512F(शून्य) { प्रत्यागम f_7_ebx[16]; }
    तार्किक AVX512DQ(शून्य) { प्रत्यागम f_7_ebx[17]; }
    तार्किक RDSEED(शून्य) { प्रत्यागम f_7_ebx[18]; }
    तार्किक ADX(शून्य) { प्रत्यागम f_7_ebx[19]; }
    तार्किक AVX512PF(शून्य) { प्रत्यागम f_7_ebx[26]; }
    तार्किक AVX512ER(शून्य) { प्रत्यागम f_7_ebx[27]; }
    तार्किक AVX512CD(शून्य) { प्रत्यागम f_7_ebx[28]; }
    तार्किक AVX512BW(शून्य) { प्रत्यागम f_7_ebx[30]; }
    तार्किक AVX512VL(शून्य) { प्रत्यागम f_7_ebx[31]; }

    तार्किक SHA(शून्य) { प्रत्यागम f_7_ebx[29]; }

    तार्किक PREFETCHWT1(शून्य) { प्रत्यागम f_7_ecx[0]; }

    तार्किक LAHF(शून्य) { प्रत्यागम f_81_ecx[0]; }
    तार्किक LZCNT(शून्य) { प्रत्यागम is_intel && f_81_ecx[5]; }
    तार्किक ABM(शून्य) { प्रत्यागम is_amd && f_81_ecx[5]; }
    तार्किक SSE4a(शून्य) { प्रत्यागम is_amd && f_81_ecx[6]; }
    तार्किक XOP(शून्य) { प्रत्यागम is_amd && f_81_ecx[11]; }
    तार्किक TBM(शून्य) { प्रत्यागम is_amd && f_81_ecx[21]; }

    तार्किक SYSCALL(शून्य) { प्रत्यागम is_intel && f_81_edx[11]; }
    तार्किक MMXEXT(शून्य) { प्रत्यागम is_amd && f_81_edx[22]; }
    तार्किक RDTSCP(शून्य) { प्रत्यागम is_intel && f_81_edx[27]; }
    तार्किक _3DNOWEXT(शून्य) { प्रत्यागम is_amd && f_81_edx[30]; }
    तार्किक _3DNOW(शून्य) { प्रत्यागम is_amd && f_81_edx[31]; }

    तार्किक AVX512_VBMI(शून्य) { प्रत्यागम f_7_ecx[1]; }
    तार्किक AVX512_VNNI(शून्य) { प्रत्यागम f_7_ecx[11]; }
    तार्किक AVX512_FP16(शून्य) { प्रत्यागम f_7_edx[23]; }
    तार्किक AVX512_BF16(शून्य) { प्रत्यागम f_7_1_eax[5]; }
    तार्किक AVX_VNNI(शून्य) { प्रत्यागम f_7_1_eax[4]; }

    तार्किक AMX_TILE(शून्य) { प्रत्यागम f_7_edx[24]; }
    तार्किक AMX_INT8(शून्य) { प्रत्यागम f_7_edx[25]; }
    तार्किक AMX_FP16(शून्य) { प्रत्यागम f_7_1_eax[21]; }
    तार्किक AMX_BF16(शून्य) { प्रत्यागम f_7_edx[22]; }

#यदि_परिभाषितम् _MSC_VER
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
#यदि_अन्तः

    cpuid_x86() {
        std::array<पूर्णाङ्क, 4> cpui;
        std::vector<std::array<पूर्णाङ्क, 4>> दत्त;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.दत्त(), 0);
        पूर्णाङ्क n_ids = cpui[0];

        प्रति (पूर्णाङ्क i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.दत्त(), i, 0);
            दत्त.push_back(cpui);
        }

        // capture vendor string
        वर्ण vendor[0x20] = {};
        *reinterpret_cast<पूर्णाङ्क *>(vendor)     = दत्त[0][1];
        *reinterpret_cast<पूर्णाङ्क *>(vendor + 4) = दत्त[0][3];
        *reinterpret_cast<पूर्णाङ्क *>(vendor + 8) = दत्त[0][2];
        एतत्->vendor = vendor;
        यदि (एतत्->vendor == "GenuineIntel") {
            is_intel = सत्य;
        } अन्यथा यदि (एतत्->vendor == "AuthenticAMD") {
            is_amd = सत्य;
        }

        // load bitset with flags for function 0x00000001
        यदि (n_ids >= 1) {
            f_1_ecx = दत्त[1][2];
            f_1_edx = दत्त[1][3];
        }

        // load bitset with flags for function 0x00000007
        यदि (n_ids >= 7) {
            f_7_ebx = दत्त[7][1];
            f_7_ecx = दत्त[7][2];
            f_7_edx = दत्त[7][3];
            cpuidex(cpui.दत्त(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.दत्त(), 0x80000000);
        अचिह्नित पूर्णाङ्क n_ex_ids = cpui[0];

        std::vector<std::array<पूर्णाङ्क, 4>> ext_दत्त;
        प्रति (अचिह्नित पूर्णाङ्क i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.दत्त(), i, 0);
            ext_दत्त.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        यदि (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_दत्त[1][2];
            f_81_edx = ext_दत्त[1][3];
        }

        // interpret CPU brand string if reported
        वर्ण brand[0x40] = {};
        यदि (n_ex_ids >= 0x80000004) {
            std::स्मृतिं_प्रतिलिख(brand, ext_दत्त[2].दत्त(), परिमाण(cpui));
            std::स्मृतिं_प्रतिलिख(brand + 16, ext_दत्त[3].दत्त(), परिमाण(cpui));
            std::स्मृतिं_प्रतिलिख(brand + 32, ext_दत्त[4].दत्त(), परिमाण(cpui));
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
    प्रारूप_लिख("CPU Vendor: %s\n", is.vendor.c_str());
    प्रारूप_लिख("Brand: %s\n", is.brand.c_str());
    प्रारूप_लिख("is_intel: %d\n", is.is_intel);
    प्रारूप_लिख("is_amd: %d\n", is.is_amd);
    प्रारूप_लिख("sse3: %d\n", is.SSE3());
    प्रारूप_लिख("pclmulqdq: %d\n", is.PCLMULQDQ());
    प्रारूप_लिख("ssse3: %d\n", is.SSSE3());
    प्रारूप_लिख("fma: %d\n", is.FMA());
    प्रारूप_लिख("cmpxchg16b: %d\n", is.CMPXCHG16B());
    प्रारूप_लिख("sse41: %d\n", is.SSE41());
    प्रारूप_लिख("sse42: %d\n", is.SSE42());
    प्रारूप_लिख("movbe: %d\n", is.MOVBE());
    प्रारूप_लिख("popcnt: %d\n", is.POPCNT());
    प्रारूप_लिख("aes: %d\n", is.AES());
    प्रारूप_लिख("xsave: %d\n", is.XSAVE());
    प्रारूप_लिख("osxsave: %d\n", is.OSXSAVE());
    प्रारूप_लिख("avx: %d\n", is.AVX());
    प्रारूप_लिख("f16c: %d\n", is.F16C());
    प्रारूप_लिख("rdrand: %d\n", is.RDRAND());
    प्रारूप_लिख("msr: %d\n", is.MSR());
    प्रारूप_लिख("cx8: %d\n", is.CX8());
    प्रारूप_लिख("sep: %d\n", is.SEP());
    प्रारूप_लिख("cmov: %d\n", is.CMOV());
    प्रारूप_लिख("clflush: %d\n", is.CLFSH());
    प्रारूप_लिख("mmx: %d\n", is.MMX());
    प्रारूप_लिख("fxsr: %d\n", is.FXSR());
    प्रारूप_लिख("sse: %d\n", is.SSE());
    प्रारूप_लिख("sse2: %d\n", is.SSE2());
    प्रारूप_लिख("fsgsbase: %d\n", is.FSGSBASE());
    प्रारूप_लिख("bmi1: %d\n", is.BMI1());
    प्रारूप_लिख("hle: %d\n", is.HLE());
    प्रारूप_लिख("avx2: %d\n", is.AVX2());
    प्रारूप_लिख("bmi2: %d\n", is.BMI2());
    प्रारूप_लिख("erms: %d\n", is.ERMS());
    प्रारूप_लिख("invpcid: %d\n", is.INVPCID());
    प्रारूप_लिख("rtm: %d\n", is.RTM());
    प्रारूप_लिख("avx512f: %d\n", is.AVX512F());
    प्रारूप_लिख("rdseed: %d\n", is.RDSEED());
    प्रारूप_लिख("adx: %d\n", is.ADX());
    प्रारूप_लिख("avx512pf: %d\n", is.AVX512PF());
    प्रारूप_लिख("avx512er: %d\n", is.AVX512ER());
    प्रारूप_लिख("avx512cd: %d\n", is.AVX512CD());
    प्रारूप_लिख("sha: %d\n", is.SHA());
    प्रारूप_लिख("prefetchwt1: %d\n", is.PREFETCHWT1());
    प्रारूप_लिख("lahf: %d\n", is.LAHF());
    प्रारूप_लिख("lzcnt: %d\n", is.LZCNT());
    प्रारूप_लिख("abm: %d\n", is.ABM());
    प्रारूप_लिख("sse4a: %d\n", is.SSE4a());
    प्रारूप_लिख("xop: %d\n", is.XOP());
    प्रारूप_लिख("tbm: %d\n", is.TBM());
    प्रारूप_लिख("syscall: %d\n", is.SYSCALL());
    प्रारूप_लिख("mmxext: %d\n", is.MMXEXT());
    प्रारूप_लिख("rdtscp: %d\n", is.RDTSCP());
    प्रारूप_लिख("3dnowext: %d\n", is._3DNOWEXT());
    प्रारूप_लिख("3dnow: %d\n", is._3DNOW());
    प्रारूप_लिख("avx512_vbmi: %d\n", is.AVX512_VBMI());
    प्रारूप_लिख("avx512_vnni: %d\n", is.AVX512_VNNI());
    प्रारूप_लिख("avx512_fp16: %d\n", is.AVX512_FP16());
    प्रारूप_लिख("avx512_bf16: %d\n", is.AVX512_BF16());
    प्रारूप_लिख("amx_tile: %d\n", is.AMX_TILE());
    प्रारूप_लिख("amx_int8: %d\n", is.AMX_INT8());
    प्रारूप_लिख("amx_fp16: %d\n", is.AMX_FP16());
    प्रारूप_लिख("amx_bf16: %d\n", is.AMX_BF16());
}
#यदि_अन्तः

स्थिर पूर्णाङ्क ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    पूर्णाङ्क score = 1;
    cpuid_x86 is;

#यदि_परिभाषितम् GGML_FMA
    यदि (!is.FMA()) { प्रत्यागम 0; }
    score += 1;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_F16C
    यदि (!is.F16C()) { प्रत्यागम 0; }
    score += 1<<1;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_SSE42
    यदि (!is.SSE42()) { प्रत्यागम 0; }
    score += 1<<2;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_BMI2
    यदि (!is.BMI2()) { प्रत्यागम 0; }
    score += 1<<3;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX
    यदि (!is.AVX()) { प्रत्यागम 0; }
    score += 1<<4;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX2
    यदि (!is.AVX2()) { प्रत्यागम 0; }
    score += 1<<5;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX_VNNI
    यदि (!is.AVX_VNNI()) { प्रत्यागम 0; }
    score += 1<<6;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX512
    यदि (!is.AVX512F()) { प्रत्यागम 0; }
    यदि (!is.AVX512CD()) { प्रत्यागम 0; }
    यदि (!is.AVX512VL()) { प्रत्यागम 0; }
    यदि (!is.AVX512DQ()) { प्रत्यागम 0; }
    यदि (!is.AVX512BW()) { प्रत्यागम 0; }
    score += 1<<7;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX512_VBMI
    यदि (!is.AVX512_VBMI()) { प्रत्यागम 0; }
    score += 1<<8;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX512_BF16
    यदि (!is.AVX512_BF16()) { प्रत्यागम 0; }
    score += 1<<9;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AVX512_VNNI
    यदि (!is.AVX512_VNNI()) { प्रत्यागम 0; }
    score += 1<<10;
#यदि_अन्तः
#यदि_परिभाषितम् GGML_AMX_INT8
    यदि (!is.AMX_INT8()) { प्रत्यागम 0; }
    score += 1<<11;
#यदि_अन्तः

    प्रत्यागम score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#यदि_अन्तः // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
