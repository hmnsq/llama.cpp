#ضمن "ggml-backend-impl.h"

#إذا defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#که_تعریف_شوی _MSC_VER
#ضمن <intrin.h>
#د_که_پای

#ضمن <cstring>
#ضمن <vector>
#ضمن <bitset>
#ضمن <array>
#ضمن <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
بنية cpuid_x86 {
    منطقي SSE3(فراغ) { أرجع f_1_ecx[0]; }
    منطقي PCLMULQDQ(فراغ) { أرجع f_1_ecx[1]; }
    منطقي MONITOR(فراغ) { أرجع f_1_ecx[3]; }
    منطقي SSSE3(فراغ) { أرجع f_1_ecx[9]; }
    منطقي FMA(فراغ) { أرجع f_1_ecx[12]; }
    منطقي CMPXCHG16B(فراغ) { أرجع f_1_ecx[13]; }
    منطقي SSE41(فراغ) { أرجع f_1_ecx[19]; }
    منطقي SSE42(فراغ) { أرجع f_1_ecx[20]; }
    منطقي MOVBE(فراغ) { أرجع f_1_ecx[22]; }
    منطقي POPCNT(فراغ) { أرجع f_1_ecx[23]; }
    منطقي AES(فراغ) { أرجع f_1_ecx[25]; }
    منطقي XSAVE(فراغ) { أرجع f_1_ecx[26]; }
    منطقي OSXSAVE(فراغ) { أرجع f_1_ecx[27]; }
    منطقي AVX(فراغ) { أرجع f_1_ecx[28]; }
    منطقي F16C(فراغ) { أرجع f_1_ecx[29]; }
    منطقي RDRAND(فراغ) { أرجع f_1_ecx[30]; }

    منطقي MSR(فراغ) { أرجع f_1_edx[5]; }
    منطقي CX8(فراغ) { أرجع f_1_edx[8]; }
    منطقي SEP(فراغ) { أرجع f_1_edx[11]; }
    منطقي CMOV(فراغ) { أرجع f_1_edx[15]; }
    منطقي CLFSH(فراغ) { أرجع f_1_edx[19]; }
    منطقي MMX(فراغ) { أرجع f_1_edx[23]; }
    منطقي FXSR(فراغ) { أرجع f_1_edx[24]; }
    منطقي SSE(فراغ) { أرجع f_1_edx[25]; }
    منطقي SSE2(فراغ) { أرجع f_1_edx[26]; }

    منطقي FSGSBASE(فراغ) { أرجع f_7_ebx[0]; }
    منطقي BMI1(فراغ) { أرجع f_7_ebx[3]; }
    منطقي HLE(فراغ) { أرجع is_intel && f_7_ebx[4]; }
    منطقي AVX2(فراغ) { أرجع f_7_ebx[5]; }
    منطقي BMI2(فراغ) { أرجع f_7_ebx[8]; }
    منطقي ERMS(فراغ) { أرجع f_7_ebx[9]; }
    منطقي INVPCID(فراغ) { أرجع f_7_ebx[10]; }
    منطقي RTM(فراغ) { أرجع is_intel && f_7_ebx[11]; }
    منطقي AVX512F(فراغ) { أرجع f_7_ebx[16]; }
    منطقي AVX512DQ(فراغ) { أرجع f_7_ebx[17]; }
    منطقي RDSEED(فراغ) { أرجع f_7_ebx[18]; }
    منطقي ADX(فراغ) { أرجع f_7_ebx[19]; }
    منطقي AVX512PF(فراغ) { أرجع f_7_ebx[26]; }
    منطقي AVX512ER(فراغ) { أرجع f_7_ebx[27]; }
    منطقي AVX512CD(فراغ) { أرجع f_7_ebx[28]; }
    منطقي AVX512BW(فراغ) { أرجع f_7_ebx[30]; }
    منطقي AVX512VL(فراغ) { أرجع f_7_ebx[31]; }

    منطقي SHA(فراغ) { أرجع f_7_ebx[29]; }

    منطقي PREFETCHWT1(فراغ) { أرجع f_7_ecx[0]; }

    منطقي LAHF(فراغ) { أرجع f_81_ecx[0]; }
    منطقي LZCNT(فراغ) { أرجع is_intel && f_81_ecx[5]; }
    منطقي ABM(فراغ) { أرجع is_amd && f_81_ecx[5]; }
    منطقي SSE4a(فراغ) { أرجع is_amd && f_81_ecx[6]; }
    منطقي XOP(فراغ) { أرجع is_amd && f_81_ecx[11]; }
    منطقي TBM(فراغ) { أرجع is_amd && f_81_ecx[21]; }

    منطقي SYSCALL(فراغ) { أرجع is_intel && f_81_edx[11]; }
    منطقي MMXEXT(فراغ) { أرجع is_amd && f_81_edx[22]; }
    منطقي RDTSCP(فراغ) { أرجع is_intel && f_81_edx[27]; }
    منطقي _3DNOWEXT(فراغ) { أرجع is_amd && f_81_edx[30]; }
    منطقي _3DNOW(فراغ) { أرجع is_amd && f_81_edx[31]; }

    منطقي AVX512_VBMI(فراغ) { أرجع f_7_ecx[1]; }
    منطقي AVX512_VNNI(فراغ) { أرجع f_7_ecx[11]; }
    منطقي AVX512_FP16(فراغ) { أرجع f_7_edx[23]; }
    منطقي AVX512_BF16(فراغ) { أرجع f_7_1_eax[5]; }
    منطقي AVX_VNNI(فراغ) { أرجع f_7_1_eax[4]; }

    منطقي AMX_TILE(فراغ) { أرجع f_7_edx[24]; }
    منطقي AMX_INT8(فراغ) { أرجع f_7_edx[25]; }
    منطقي AMX_FP16(فراغ) { أرجع f_7_1_eax[21]; }
    منطقي AMX_BF16(فراغ) { أرجع f_7_edx[22]; }

#که_تعریف_شوی _MSC_VER
    ساكن فراغ cpuid(صحيح cpu_info[4], صحيح eax) {
        __cpuid(cpu_info, eax);
    }
    ساكن فراغ cpuidex(صحيح cpu_info[4], صحيح eax, صحيح ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#وإلا
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
#د_که_پای

    cpuid_x86() {
        std::array<صحيح, 4> cpui;
        std::vector<std::array<صحيح, 4>> معلومات;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.معلومات(), 0);
        صحيح n_ids = cpui[0];

        لكل (صحيح i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.معلومات(), i, 0);
            معلومات.push_back(cpui);
        }

        // capture vendor string
        حرف vendor[0x20] = {};
        *reinterpret_cast<صحيح *>(vendor)     = معلومات[0][1];
        *reinterpret_cast<صحيح *>(vendor + 4) = معلومات[0][3];
        *reinterpret_cast<صحيح *>(vendor + 8) = معلومات[0][2];
        هذا->vendor = vendor;
        إذا (هذا->vendor == "GenuineIntel") {
            is_intel = ریښتیا;
        } وإلا إذا (هذا->vendor == "AuthenticAMD") {
            is_amd = ریښتیا;
        }

        // load bitset with flags for function 0x00000001
        إذا (n_ids >= 1) {
            f_1_ecx = معلومات[1][2];
            f_1_edx = معلومات[1][3];
        }

        // load bitset with flags for function 0x00000007
        إذا (n_ids >= 7) {
            f_7_ebx = معلومات[7][1];
            f_7_ecx = معلومات[7][2];
            f_7_edx = معلومات[7][3];
            cpuidex(cpui.معلومات(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.معلومات(), 0x80000000);
        بینښه صحيح n_ex_ids = cpui[0];

        std::vector<std::array<صحيح, 4>> ext_معلومات;
        لكل (بینښه صحيح i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.معلومات(), i, 0);
            ext_معلومات.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        إذا (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_معلومات[1][2];
            f_81_edx = ext_معلومات[1][3];
        }

        // interpret CPU brand string if reported
        حرف brand[0x40] = {};
        إذا (n_ex_ids >= 0x80000004) {
            std::حافظه_نقل(brand, ext_معلومات[2].معلومات(), حجم(cpui));
            std::حافظه_نقل(brand + 16, ext_معلومات[3].معلومات(), حجم(cpui));
            std::حافظه_نقل(brand + 32, ext_معلومات[4].معلومات(), حجم(cpui));
            هذا->brand = brand;
        }
    }

    منطقي is_intel = خطأ;
    منطقي is_amd = خطأ;
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

#إذا 0
فراغ test_x86_is() {
    cpuid_x86 is;
    چاپ_کړه("CPU Vendor: %s\n", is.vendor.c_str());
    چاپ_کړه("Brand: %s\n", is.brand.c_str());
    چاپ_کړه("is_intel: %d\n", is.is_intel);
    چاپ_کړه("is_amd: %d\n", is.is_amd);
    چاپ_کړه("sse3: %d\n", is.SSE3());
    چاپ_کړه("pclmulqdq: %d\n", is.PCLMULQDQ());
    چاپ_کړه("ssse3: %d\n", is.SSSE3());
    چاپ_کړه("fma: %d\n", is.FMA());
    چاپ_کړه("cmpxchg16b: %d\n", is.CMPXCHG16B());
    چاپ_کړه("sse41: %d\n", is.SSE41());
    چاپ_کړه("sse42: %d\n", is.SSE42());
    چاپ_کړه("movbe: %d\n", is.MOVBE());
    چاپ_کړه("popcnt: %d\n", is.POPCNT());
    چاپ_کړه("aes: %d\n", is.AES());
    چاپ_کړه("xsave: %d\n", is.XSAVE());
    چاپ_کړه("osxsave: %d\n", is.OSXSAVE());
    چاپ_کړه("avx: %d\n", is.AVX());
    چاپ_کړه("f16c: %d\n", is.F16C());
    چاپ_کړه("rdrand: %d\n", is.RDRAND());
    چاپ_کړه("msr: %d\n", is.MSR());
    چاپ_کړه("cx8: %d\n", is.CX8());
    چاپ_کړه("sep: %d\n", is.SEP());
    چاپ_کړه("cmov: %d\n", is.CMOV());
    چاپ_کړه("clflush: %d\n", is.CLFSH());
    چاپ_کړه("mmx: %d\n", is.MMX());
    چاپ_کړه("fxsr: %d\n", is.FXSR());
    چاپ_کړه("sse: %d\n", is.SSE());
    چاپ_کړه("sse2: %d\n", is.SSE2());
    چاپ_کړه("fsgsbase: %d\n", is.FSGSBASE());
    چاپ_کړه("bmi1: %d\n", is.BMI1());
    چاپ_کړه("hle: %d\n", is.HLE());
    چاپ_کړه("avx2: %d\n", is.AVX2());
    چاپ_کړه("bmi2: %d\n", is.BMI2());
    چاپ_کړه("erms: %d\n", is.ERMS());
    چاپ_کړه("invpcid: %d\n", is.INVPCID());
    چاپ_کړه("rtm: %d\n", is.RTM());
    چاپ_کړه("avx512f: %d\n", is.AVX512F());
    چاپ_کړه("rdseed: %d\n", is.RDSEED());
    چاپ_کړه("adx: %d\n", is.ADX());
    چاپ_کړه("avx512pf: %d\n", is.AVX512PF());
    چاپ_کړه("avx512er: %d\n", is.AVX512ER());
    چاپ_کړه("avx512cd: %d\n", is.AVX512CD());
    چاپ_کړه("sha: %d\n", is.SHA());
    چاپ_کړه("prefetchwt1: %d\n", is.PREFETCHWT1());
    چاپ_کړه("lahf: %d\n", is.LAHF());
    چاپ_کړه("lzcnt: %d\n", is.LZCNT());
    چاپ_کړه("abm: %d\n", is.ABM());
    چاپ_کړه("sse4a: %d\n", is.SSE4a());
    چاپ_کړه("xop: %d\n", is.XOP());
    چاپ_کړه("tbm: %d\n", is.TBM());
    چاپ_کړه("syscall: %d\n", is.SYSCALL());
    چاپ_کړه("mmxext: %d\n", is.MMXEXT());
    چاپ_کړه("rdtscp: %d\n", is.RDTSCP());
    چاپ_کړه("3dnowext: %d\n", is._3DNOWEXT());
    چاپ_کړه("3dnow: %d\n", is._3DNOW());
    چاپ_کړه("avx512_vbmi: %d\n", is.AVX512_VBMI());
    چاپ_کړه("avx512_vnni: %d\n", is.AVX512_VNNI());
    چاپ_کړه("avx512_fp16: %d\n", is.AVX512_FP16());
    چاپ_کړه("avx512_bf16: %d\n", is.AVX512_BF16());
    چاپ_کړه("amx_tile: %d\n", is.AMX_TILE());
    چاپ_کړه("amx_int8: %d\n", is.AMX_INT8());
    چاپ_کړه("amx_fp16: %d\n", is.AMX_FP16());
    چاپ_کړه("amx_bf16: %d\n", is.AMX_BF16());
}
#د_که_پای

ساكن صحيح ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    صحيح score = 1;
    cpuid_x86 is;

#که_تعریف_شوی GGML_FMA
    إذا (!is.FMA()) { أرجع 0; }
    score += 1;
#د_که_پای
#که_تعریف_شوی GGML_F16C
    إذا (!is.F16C()) { أرجع 0; }
    score += 1<<1;
#د_که_پای
#که_تعریف_شوی GGML_SSE42
    إذا (!is.SSE42()) { أرجع 0; }
    score += 1<<2;
#د_که_پای
#که_تعریف_شوی GGML_BMI2
    إذا (!is.BMI2()) { أرجع 0; }
    score += 1<<3;
#د_که_پای
#که_تعریف_شوی GGML_AVX
    إذا (!is.AVX()) { أرجع 0; }
    score += 1<<4;
#د_که_پای
#که_تعریف_شوی GGML_AVX2
    إذا (!is.AVX2()) { أرجع 0; }
    score += 1<<5;
#د_که_پای
#که_تعریف_شوی GGML_AVX_VNNI
    إذا (!is.AVX_VNNI()) { أرجع 0; }
    score += 1<<6;
#د_که_پای
#که_تعریف_شوی GGML_AVX512
    إذا (!is.AVX512F()) { أرجع 0; }
    إذا (!is.AVX512CD()) { أرجع 0; }
    إذا (!is.AVX512VL()) { أرجع 0; }
    إذا (!is.AVX512DQ()) { أرجع 0; }
    إذا (!is.AVX512BW()) { أرجع 0; }
    score += 1<<7;
#د_که_پای
#که_تعریف_شوی GGML_AVX512_VBMI
    إذا (!is.AVX512_VBMI()) { أرجع 0; }
    score += 1<<8;
#د_که_پای
#که_تعریف_شوی GGML_AVX512_BF16
    إذا (!is.AVX512_BF16()) { أرجع 0; }
    score += 1<<9;
#د_که_پای
#که_تعریف_شوی GGML_AVX512_VNNI
    إذا (!is.AVX512_VNNI()) { أرجع 0; }
    score += 1<<10;
#د_که_پای
#که_تعریف_شوی GGML_AMX_INT8
    إذا (!is.AMX_INT8()) { أرجع 0; }
    score += 1<<11;
#د_که_پای

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#د_که_پای // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
