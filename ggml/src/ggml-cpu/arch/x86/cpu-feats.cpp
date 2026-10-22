#అన్తర్భావ "ggml-backend-impl.h"

#ఒకవేళ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#నిర్వచితమైతే _MSC_VER
#అన్తర్భావ <intrin.h>
#అయితే_ముగింపు

#అన్తర్భావ <cstring>
#అన్తర్భావ <vector>
#అన్తర్భావ <bitset>
#అన్తర్భావ <array>
#అన్తర్భావ <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
నిర్మాణం cpuid_x86 {
    తార్కిక SSE3(శూన్య) { తిరిగి f_1_ecx[0]; }
    తార్కిక PCLMULQDQ(శూన్య) { తిరిగి f_1_ecx[1]; }
    తార్కిక MONITOR(శూన్య) { తిరిగి f_1_ecx[3]; }
    తార్కిక SSSE3(శూన్య) { తిరిగి f_1_ecx[9]; }
    తార్కిక FMA(శూన్య) { తిరిగి f_1_ecx[12]; }
    తార్కిక CMPXCHG16B(శూన్య) { తిరిగి f_1_ecx[13]; }
    తార్కిక SSE41(శూన్య) { తిరిగి f_1_ecx[19]; }
    తార్కిక SSE42(శూన్య) { తిరిగి f_1_ecx[20]; }
    తార్కిక MOVBE(శూన్య) { తిరిగి f_1_ecx[22]; }
    తార్కిక POPCNT(శూన్య) { తిరిగి f_1_ecx[23]; }
    తార్కిక AES(శూన్య) { తిరిగి f_1_ecx[25]; }
    తార్కిక XSAVE(శూన్య) { తిరిగి f_1_ecx[26]; }
    తార్కిక OSXSAVE(శూన్య) { తిరిగి f_1_ecx[27]; }
    తార్కిక AVX(శూన్య) { తిరిగి f_1_ecx[28]; }
    తార్కిక F16C(శూన్య) { తిరిగి f_1_ecx[29]; }
    తార్కిక RDRAND(శూన్య) { తిరిగి f_1_ecx[30]; }

    తార్కిక MSR(శూన్య) { తిరిగి f_1_edx[5]; }
    తార్కిక CX8(శూన్య) { తిరిగి f_1_edx[8]; }
    తార్కిక SEP(శూన్య) { తిరిగి f_1_edx[11]; }
    తార్కిక CMOV(శూన్య) { తిరిగి f_1_edx[15]; }
    తార్కిక CLFSH(శూన్య) { తిరిగి f_1_edx[19]; }
    తార్కిక MMX(శూన్య) { తిరిగి f_1_edx[23]; }
    తార్కిక FXSR(శూన్య) { తిరిగి f_1_edx[24]; }
    తార్కిక SSE(శూన్య) { తిరిగి f_1_edx[25]; }
    తార్కిక SSE2(శూన్య) { తిరిగి f_1_edx[26]; }

    తార్కిక FSGSBASE(శూన్య) { తిరిగి f_7_ebx[0]; }
    తార్కిక BMI1(శూన్య) { తిరిగి f_7_ebx[3]; }
    తార్కిక HLE(శూన్య) { తిరిగి is_intel && f_7_ebx[4]; }
    తార్కిక AVX2(శూన్య) { తిరిగి f_7_ebx[5]; }
    తార్కిక BMI2(శూన్య) { తిరిగి f_7_ebx[8]; }
    తార్కిక ERMS(శూన్య) { తిరిగి f_7_ebx[9]; }
    తార్కిక INVPCID(శూన్య) { తిరిగి f_7_ebx[10]; }
    తార్కిక RTM(శూన్య) { తిరిగి is_intel && f_7_ebx[11]; }
    తార్కిక AVX512F(శూన్య) { తిరిగి f_7_ebx[16]; }
    తార్కిక AVX512DQ(శూన్య) { తిరిగి f_7_ebx[17]; }
    తార్కిక RDSEED(శూన్య) { తిరిగి f_7_ebx[18]; }
    తార్కిక ADX(శూన్య) { తిరిగి f_7_ebx[19]; }
    తార్కిక AVX512PF(శూన్య) { తిరిగి f_7_ebx[26]; }
    తార్కిక AVX512ER(శూన్య) { తిరిగి f_7_ebx[27]; }
    తార్కిక AVX512CD(శూన్య) { తిరిగి f_7_ebx[28]; }
    తార్కిక AVX512BW(శూన్య) { తిరిగి f_7_ebx[30]; }
    తార్కిక AVX512VL(శూన్య) { తిరిగి f_7_ebx[31]; }

    తార్కిక SHA(శూన్య) { తిరిగి f_7_ebx[29]; }

    తార్కిక PREFETCHWT1(శూన్య) { తిరిగి f_7_ecx[0]; }

    తార్కిక LAHF(శూన్య) { తిరిగి f_81_ecx[0]; }
    తార్కిక LZCNT(శూన్య) { తిరిగి is_intel && f_81_ecx[5]; }
    తార్కిక ABM(శూన్య) { తిరిగి is_amd && f_81_ecx[5]; }
    తార్కిక SSE4a(శూన్య) { తిరిగి is_amd && f_81_ecx[6]; }
    తార్కిక XOP(శూన్య) { తిరిగి is_amd && f_81_ecx[11]; }
    తార్కిక TBM(శూన్య) { తిరిగి is_amd && f_81_ecx[21]; }

    తార్కిక SYSCALL(శూన్య) { తిరిగి is_intel && f_81_edx[11]; }
    తార్కిక MMXEXT(శూన్య) { తిరిగి is_amd && f_81_edx[22]; }
    తార్కిక RDTSCP(శూన్య) { తిరిగి is_intel && f_81_edx[27]; }
    తార్కిక _3DNOWEXT(శూన్య) { తిరిగి is_amd && f_81_edx[30]; }
    తార్కిక _3DNOW(శూన్య) { తిరిగి is_amd && f_81_edx[31]; }

    తార్కిక AVX512_VBMI(శూన్య) { తిరిగి f_7_ecx[1]; }
    తార్కిక AVX512_VNNI(శూన్య) { తిరిగి f_7_ecx[11]; }
    తార్కిక AVX512_FP16(శూన్య) { తిరిగి f_7_edx[23]; }
    తార్కిక AVX512_BF16(శూన్య) { తిరిగి f_7_1_eax[5]; }
    తార్కిక AVX_VNNI(శూన్య) { తిరిగి f_7_1_eax[4]; }

    తార్కిక AMX_TILE(శూన్య) { తిరిగి f_7_edx[24]; }
    తార్కిక AMX_INT8(శూన్య) { తిరిగి f_7_edx[25]; }
    తార్కిక AMX_FP16(శూన్య) { తిరిగి f_7_1_eax[21]; }
    తార్కిక AMX_BF16(శూన్య) { తిరిగి f_7_edx[22]; }

#నిర్వచితమైతే _MSC_VER
    స్థిర శూన్య cpuid(పూర్ణాంకం cpu_info[4], పూర్ణాంకం eax) {
        __cpuid(cpu_info, eax);
    }
    స్థిర శూన్య cpuidex(పూర్ణాంకం cpu_info[4], పూర్ణాంకం eax, పూర్ణాంకం ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#లేకపోతే
    స్థిర శూన్య cpuid(పూర్ణాంకం cpu_info[4], పూర్ణాంకం eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    స్థిర శూన్య cpuidex(పూర్ణాంకం cpu_info[4], పూర్ణాంకం eax, పూర్ణాంకం ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#అయితే_ముగింపు

    cpuid_x86() {
        std::array<పూర్ణాంకం, 4> cpui;
        std::vector<std::array<పూర్ణాంకం, 4>> దత్తాంశం;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.దత్తాంశం(), 0);
        పూర్ణాంకం n_ids = cpui[0];

        కొరకు (పూర్ణాంకం i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.దత్తాంశం(), i, 0);
            దత్తాంశం.push_back(cpui);
        }

        // capture vendor string
        అక్షరం vendor[0x20] = {};
        *reinterpret_cast<పూర్ణాంకం *>(vendor)     = దత్తాంశం[0][1];
        *reinterpret_cast<పూర్ణాంకం *>(vendor + 4) = దత్తాంశం[0][3];
        *reinterpret_cast<పూర్ణాంకం *>(vendor + 8) = దత్తాంశం[0][2];
        ఇది->vendor = vendor;
        ఒకవేళ (ఇది->vendor == "GenuineIntel") {
            is_intel = నిజం;
        } లేకపోతే ఒకవేళ (ఇది->vendor == "AuthenticAMD") {
            is_amd = నిజం;
        }

        // load bitset with flags for function 0x00000001
        ఒకవేళ (n_ids >= 1) {
            f_1_ecx = దత్తాంశం[1][2];
            f_1_edx = దత్తాంశం[1][3];
        }

        // load bitset with flags for function 0x00000007
        ఒకవేళ (n_ids >= 7) {
            f_7_ebx = దత్తాంశం[7][1];
            f_7_ecx = దత్తాంశం[7][2];
            f_7_edx = దత్తాంశం[7][3];
            cpuidex(cpui.దత్తాంశం(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.దత్తాంశం(), 0x80000000);
        గుర్తులేని పూర్ణాంకం n_ex_ids = cpui[0];

        std::vector<std::array<పూర్ణాంకం, 4>> ext_దత్తాంశం;
        కొరకు (గుర్తులేని పూర్ణాంకం i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.దత్తాంశం(), i, 0);
            ext_దత్తాంశం.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ఒకవేళ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_దత్తాంశం[1][2];
            f_81_edx = ext_దత్తాంశం[1][3];
        }

        // interpret CPU brand string if reported
        అక్షరం brand[0x40] = {};
        ఒకవేళ (n_ex_ids >= 0x80000004) {
            std::జ్ఞాపకం_నకలు(brand, ext_దత్తాంశం[2].దత్తాంశం(), పరిమాణం(cpui));
            std::జ్ఞాపకం_నకలు(brand + 16, ext_దత్తాంశం[3].దత్తాంశం(), పరిమాణం(cpui));
            std::జ్ఞాపకం_నకలు(brand + 32, ext_దత్తాంశం[4].దత్తాంశం(), పరిమాణం(cpui));
            ఇది->brand = brand;
        }
    }

    తార్కిక is_intel = అబద్ధం;
    తార్కిక is_amd = అబద్ధం;
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

#ఒకవేళ 0
శూన్య test_x86_is() {
    cpuid_x86 is;
    రాయి("CPU Vendor: %s\n", is.vendor.c_str());
    రాయి("Brand: %s\n", is.brand.c_str());
    రాయి("is_intel: %d\n", is.is_intel);
    రాయి("is_amd: %d\n", is.is_amd);
    రాయి("sse3: %d\n", is.SSE3());
    రాయి("pclmulqdq: %d\n", is.PCLMULQDQ());
    రాయి("ssse3: %d\n", is.SSSE3());
    రాయి("fma: %d\n", is.FMA());
    రాయి("cmpxchg16b: %d\n", is.CMPXCHG16B());
    రాయి("sse41: %d\n", is.SSE41());
    రాయి("sse42: %d\n", is.SSE42());
    రాయి("movbe: %d\n", is.MOVBE());
    రాయి("popcnt: %d\n", is.POPCNT());
    రాయి("aes: %d\n", is.AES());
    రాయి("xsave: %d\n", is.XSAVE());
    రాయి("osxsave: %d\n", is.OSXSAVE());
    రాయి("avx: %d\n", is.AVX());
    రాయి("f16c: %d\n", is.F16C());
    రాయి("rdrand: %d\n", is.RDRAND());
    రాయి("msr: %d\n", is.MSR());
    రాయి("cx8: %d\n", is.CX8());
    రాయి("sep: %d\n", is.SEP());
    రాయి("cmov: %d\n", is.CMOV());
    రాయి("clflush: %d\n", is.CLFSH());
    రాయి("mmx: %d\n", is.MMX());
    రాయి("fxsr: %d\n", is.FXSR());
    రాయి("sse: %d\n", is.SSE());
    రాయి("sse2: %d\n", is.SSE2());
    రాయి("fsgsbase: %d\n", is.FSGSBASE());
    రాయి("bmi1: %d\n", is.BMI1());
    రాయి("hle: %d\n", is.HLE());
    రాయి("avx2: %d\n", is.AVX2());
    రాయి("bmi2: %d\n", is.BMI2());
    రాయి("erms: %d\n", is.ERMS());
    రాయి("invpcid: %d\n", is.INVPCID());
    రాయి("rtm: %d\n", is.RTM());
    రాయి("avx512f: %d\n", is.AVX512F());
    రాయి("rdseed: %d\n", is.RDSEED());
    రాయి("adx: %d\n", is.ADX());
    రాయి("avx512pf: %d\n", is.AVX512PF());
    రాయి("avx512er: %d\n", is.AVX512ER());
    రాయి("avx512cd: %d\n", is.AVX512CD());
    రాయి("sha: %d\n", is.SHA());
    రాయి("prefetchwt1: %d\n", is.PREFETCHWT1());
    రాయి("lahf: %d\n", is.LAHF());
    రాయి("lzcnt: %d\n", is.LZCNT());
    రాయి("abm: %d\n", is.ABM());
    రాయి("sse4a: %d\n", is.SSE4a());
    రాయి("xop: %d\n", is.XOP());
    రాయి("tbm: %d\n", is.TBM());
    రాయి("syscall: %d\n", is.SYSCALL());
    రాయి("mmxext: %d\n", is.MMXEXT());
    రాయి("rdtscp: %d\n", is.RDTSCP());
    రాయి("3dnowext: %d\n", is._3DNOWEXT());
    రాయి("3dnow: %d\n", is._3DNOW());
    రాయి("avx512_vbmi: %d\n", is.AVX512_VBMI());
    రాయి("avx512_vnni: %d\n", is.AVX512_VNNI());
    రాయి("avx512_fp16: %d\n", is.AVX512_FP16());
    రాయి("avx512_bf16: %d\n", is.AVX512_BF16());
    రాయి("amx_tile: %d\n", is.AMX_TILE());
    రాయి("amx_int8: %d\n", is.AMX_INT8());
    రాయి("amx_fp16: %d\n", is.AMX_FP16());
    రాయి("amx_bf16: %d\n", is.AMX_BF16());
}
#అయితే_ముగింపు

స్థిర పూర్ణాంకం ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    పూర్ణాంకం score = 1;
    cpuid_x86 is;

#నిర్వచితమైతే GGML_FMA
    ఒకవేళ (!is.FMA()) { తిరిగి 0; }
    score += 1;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_F16C
    ఒకవేళ (!is.F16C()) { తిరిగి 0; }
    score += 1<<1;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_SSE42
    ఒకవేళ (!is.SSE42()) { తిరిగి 0; }
    score += 1<<2;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_BMI2
    ఒకవేళ (!is.BMI2()) { తిరిగి 0; }
    score += 1<<3;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX
    ఒకవేళ (!is.AVX()) { తిరిగి 0; }
    score += 1<<4;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX2
    ఒకవేళ (!is.AVX2()) { తిరిగి 0; }
    score += 1<<5;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX_VNNI
    ఒకవేళ (!is.AVX_VNNI()) { తిరిగి 0; }
    score += 1<<6;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX512
    ఒకవేళ (!is.AVX512F()) { తిరిగి 0; }
    ఒకవేళ (!is.AVX512CD()) { తిరిగి 0; }
    ఒకవేళ (!is.AVX512VL()) { తిరిగి 0; }
    ఒకవేళ (!is.AVX512DQ()) { తిరిగి 0; }
    ఒకవేళ (!is.AVX512BW()) { తిరిగి 0; }
    score += 1<<7;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX512_VBMI
    ఒకవేళ (!is.AVX512_VBMI()) { తిరిగి 0; }
    score += 1<<8;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX512_BF16
    ఒకవేళ (!is.AVX512_BF16()) { తిరిగి 0; }
    score += 1<<9;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AVX512_VNNI
    ఒకవేళ (!is.AVX512_VNNI()) { తిరిగి 0; }
    score += 1<<10;
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_AMX_INT8
    ఒకవేళ (!is.AMX_INT8()) { తిరిగి 0; }
    score += 1<<11;
#అయితే_ముగింపు

    తిరిగి score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#అయితే_ముగింపు // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
