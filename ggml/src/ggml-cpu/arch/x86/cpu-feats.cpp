#ضمن "ggml-backend-impl.h"

#إذا defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#اگر_تعریف_شده _MSC_VER
#ضمن <intrin.h>
#پایان_اگر

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

#اگر_تعریف_شده _MSC_VER
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
#پایان_اگر

    cpuid_x86() {
        std::array<صحيح, 4> cpui;
        std::vector<std::array<صحيح, 4>> داده;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.داده(), 0);
        صحيح n_ids = cpui[0];

        لكل (صحيح i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.داده(), i, 0);
            داده.push_back(cpui);
        }

        // capture vendor string
        حرف vendor[0x20] = {};
        *reinterpret_cast<صحيح *>(vendor)     = داده[0][1];
        *reinterpret_cast<صحيح *>(vendor + 4) = داده[0][3];
        *reinterpret_cast<صحيح *>(vendor + 8) = داده[0][2];
        هذا->vendor = vendor;
        إذا (هذا->vendor == "GenuineIntel") {
            is_intel = درست;
        } وإلا إذا (هذا->vendor == "AuthenticAMD") {
            is_amd = درست;
        }

        // load bitset with flags for function 0x00000001
        إذا (n_ids >= 1) {
            f_1_ecx = داده[1][2];
            f_1_edx = داده[1][3];
        }

        // load bitset with flags for function 0x00000007
        إذا (n_ids >= 7) {
            f_7_ebx = داده[7][1];
            f_7_ecx = داده[7][2];
            f_7_edx = داده[7][3];
            cpuidex(cpui.داده(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.داده(), 0x80000000);
        بیعلامت صحيح n_ex_ids = cpui[0];

        std::vector<std::array<صحيح, 4>> ext_داده;
        لكل (بیعلامت صحيح i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.داده(), i, 0);
            ext_داده.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        إذا (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_داده[1][2];
            f_81_edx = ext_داده[1][3];
        }

        // interpret CPU brand string if reported
        حرف brand[0x40] = {};
        إذا (n_ex_ids >= 0x80000004) {
            std::کپی_حافظه(brand, ext_داده[2].داده(), حجم(cpui));
            std::کپی_حافظه(brand + 16, ext_داده[3].داده(), حجم(cpui));
            std::کپی_حافظه(brand + 32, ext_داده[4].داده(), حجم(cpui));
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
    چاپ_کن("CPU Vendor: %s\n", is.vendor.c_str());
    چاپ_کن("Brand: %s\n", is.brand.c_str());
    چاپ_کن("is_intel: %d\n", is.is_intel);
    چاپ_کن("is_amd: %d\n", is.is_amd);
    چاپ_کن("sse3: %d\n", is.SSE3());
    چاپ_کن("pclmulqdq: %d\n", is.PCLMULQDQ());
    چاپ_کن("ssse3: %d\n", is.SSSE3());
    چاپ_کن("fma: %d\n", is.FMA());
    چاپ_کن("cmpxchg16b: %d\n", is.CMPXCHG16B());
    چاپ_کن("sse41: %d\n", is.SSE41());
    چاپ_کن("sse42: %d\n", is.SSE42());
    چاپ_کن("movbe: %d\n", is.MOVBE());
    چاپ_کن("popcnt: %d\n", is.POPCNT());
    چاپ_کن("aes: %d\n", is.AES());
    چاپ_کن("xsave: %d\n", is.XSAVE());
    چاپ_کن("osxsave: %d\n", is.OSXSAVE());
    چاپ_کن("avx: %d\n", is.AVX());
    چاپ_کن("f16c: %d\n", is.F16C());
    چاپ_کن("rdrand: %d\n", is.RDRAND());
    چاپ_کن("msr: %d\n", is.MSR());
    چاپ_کن("cx8: %d\n", is.CX8());
    چاپ_کن("sep: %d\n", is.SEP());
    چاپ_کن("cmov: %d\n", is.CMOV());
    چاپ_کن("clflush: %d\n", is.CLFSH());
    چاپ_کن("mmx: %d\n", is.MMX());
    چاپ_کن("fxsr: %d\n", is.FXSR());
    چاپ_کن("sse: %d\n", is.SSE());
    چاپ_کن("sse2: %d\n", is.SSE2());
    چاپ_کن("fsgsbase: %d\n", is.FSGSBASE());
    چاپ_کن("bmi1: %d\n", is.BMI1());
    چاپ_کن("hle: %d\n", is.HLE());
    چاپ_کن("avx2: %d\n", is.AVX2());
    چاپ_کن("bmi2: %d\n", is.BMI2());
    چاپ_کن("erms: %d\n", is.ERMS());
    چاپ_کن("invpcid: %d\n", is.INVPCID());
    چاپ_کن("rtm: %d\n", is.RTM());
    چاپ_کن("avx512f: %d\n", is.AVX512F());
    چاپ_کن("rdseed: %d\n", is.RDSEED());
    چاپ_کن("adx: %d\n", is.ADX());
    چاپ_کن("avx512pf: %d\n", is.AVX512PF());
    چاپ_کن("avx512er: %d\n", is.AVX512ER());
    چاپ_کن("avx512cd: %d\n", is.AVX512CD());
    چاپ_کن("sha: %d\n", is.SHA());
    چاپ_کن("prefetchwt1: %d\n", is.PREFETCHWT1());
    چاپ_کن("lahf: %d\n", is.LAHF());
    چاپ_کن("lzcnt: %d\n", is.LZCNT());
    چاپ_کن("abm: %d\n", is.ABM());
    چاپ_کن("sse4a: %d\n", is.SSE4a());
    چاپ_کن("xop: %d\n", is.XOP());
    چاپ_کن("tbm: %d\n", is.TBM());
    چاپ_کن("syscall: %d\n", is.SYSCALL());
    چاپ_کن("mmxext: %d\n", is.MMXEXT());
    چاپ_کن("rdtscp: %d\n", is.RDTSCP());
    چاپ_کن("3dnowext: %d\n", is._3DNOWEXT());
    چاپ_کن("3dnow: %d\n", is._3DNOW());
    چاپ_کن("avx512_vbmi: %d\n", is.AVX512_VBMI());
    چاپ_کن("avx512_vnni: %d\n", is.AVX512_VNNI());
    چاپ_کن("avx512_fp16: %d\n", is.AVX512_FP16());
    چاپ_کن("avx512_bf16: %d\n", is.AVX512_BF16());
    چاپ_کن("amx_tile: %d\n", is.AMX_TILE());
    چاپ_کن("amx_int8: %d\n", is.AMX_INT8());
    چاپ_کن("amx_fp16: %d\n", is.AMX_FP16());
    چاپ_کن("amx_bf16: %d\n", is.AMX_BF16());
}
#پایان_اگر

ساكن صحيح ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    صحيح score = 1;
    cpuid_x86 is;

#اگر_تعریف_شده GGML_FMA
    إذا (!is.FMA()) { أرجع 0; }
    score += 1;
#پایان_اگر
#اگر_تعریف_شده GGML_F16C
    إذا (!is.F16C()) { أرجع 0; }
    score += 1<<1;
#پایان_اگر
#اگر_تعریف_شده GGML_SSE42
    إذا (!is.SSE42()) { أرجع 0; }
    score += 1<<2;
#پایان_اگر
#اگر_تعریف_شده GGML_BMI2
    إذا (!is.BMI2()) { أرجع 0; }
    score += 1<<3;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX
    إذا (!is.AVX()) { أرجع 0; }
    score += 1<<4;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX2
    إذا (!is.AVX2()) { أرجع 0; }
    score += 1<<5;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX_VNNI
    إذا (!is.AVX_VNNI()) { أرجع 0; }
    score += 1<<6;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX512
    إذا (!is.AVX512F()) { أرجع 0; }
    إذا (!is.AVX512CD()) { أرجع 0; }
    إذا (!is.AVX512VL()) { أرجع 0; }
    إذا (!is.AVX512DQ()) { أرجع 0; }
    إذا (!is.AVX512BW()) { أرجع 0; }
    score += 1<<7;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX512_VBMI
    إذا (!is.AVX512_VBMI()) { أرجع 0; }
    score += 1<<8;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX512_BF16
    إذا (!is.AVX512_BF16()) { أرجع 0; }
    score += 1<<9;
#پایان_اگر
#اگر_تعریف_شده GGML_AVX512_VNNI
    إذا (!is.AVX512_VNNI()) { أرجع 0; }
    score += 1<<10;
#پایان_اگر
#اگر_تعریف_شده GGML_AMX_INT8
    إذا (!is.AMX_INT8()) { أرجع 0; }
    score += 1<<11;
#پایان_اگر

    أرجع score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#پایان_اگر // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
