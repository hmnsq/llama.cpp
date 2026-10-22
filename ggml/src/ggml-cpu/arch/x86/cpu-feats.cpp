#شامل "ggml-backend-impl.h"

#اگر defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#اگر_تعریف_شدہ _MSC_VER
#شامل <intrin.h>
#اگر_ختم

#شامل <cstring>
#شامل <vector>
#شامل <bitset>
#شامل <array>
#شامل <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
ساخت cpuid_x86 {
    منطقی SSE3(خالی) { واپس f_1_ecx[0]; }
    منطقی PCLMULQDQ(خالی) { واپس f_1_ecx[1]; }
    منطقی MONITOR(خالی) { واپس f_1_ecx[3]; }
    منطقی SSSE3(خالی) { واپس f_1_ecx[9]; }
    منطقی FMA(خالی) { واپس f_1_ecx[12]; }
    منطقی CMPXCHG16B(خالی) { واپس f_1_ecx[13]; }
    منطقی SSE41(خالی) { واپس f_1_ecx[19]; }
    منطقی SSE42(خالی) { واپس f_1_ecx[20]; }
    منطقی MOVBE(خالی) { واپس f_1_ecx[22]; }
    منطقی POPCNT(خالی) { واپس f_1_ecx[23]; }
    منطقی AES(خالی) { واپس f_1_ecx[25]; }
    منطقی XSAVE(خالی) { واپس f_1_ecx[26]; }
    منطقی OSXSAVE(خالی) { واپس f_1_ecx[27]; }
    منطقی AVX(خالی) { واپس f_1_ecx[28]; }
    منطقی F16C(خالی) { واپس f_1_ecx[29]; }
    منطقی RDRAND(خالی) { واپس f_1_ecx[30]; }

    منطقی MSR(خالی) { واپس f_1_edx[5]; }
    منطقی CX8(خالی) { واپس f_1_edx[8]; }
    منطقی SEP(خالی) { واپس f_1_edx[11]; }
    منطقی CMOV(خالی) { واپس f_1_edx[15]; }
    منطقی CLFSH(خالی) { واپس f_1_edx[19]; }
    منطقی MMX(خالی) { واپس f_1_edx[23]; }
    منطقی FXSR(خالی) { واپس f_1_edx[24]; }
    منطقی SSE(خالی) { واپس f_1_edx[25]; }
    منطقی SSE2(خالی) { واپس f_1_edx[26]; }

    منطقی FSGSBASE(خالی) { واپس f_7_ebx[0]; }
    منطقی BMI1(خالی) { واپس f_7_ebx[3]; }
    منطقی HLE(خالی) { واپس is_intel && f_7_ebx[4]; }
    منطقی AVX2(خالی) { واپس f_7_ebx[5]; }
    منطقی BMI2(خالی) { واپس f_7_ebx[8]; }
    منطقی ERMS(خالی) { واپس f_7_ebx[9]; }
    منطقی INVPCID(خالی) { واپس f_7_ebx[10]; }
    منطقی RTM(خالی) { واپس is_intel && f_7_ebx[11]; }
    منطقی AVX512F(خالی) { واپس f_7_ebx[16]; }
    منطقی AVX512DQ(خالی) { واپس f_7_ebx[17]; }
    منطقی RDSEED(خالی) { واپس f_7_ebx[18]; }
    منطقی ADX(خالی) { واپس f_7_ebx[19]; }
    منطقی AVX512PF(خالی) { واپس f_7_ebx[26]; }
    منطقی AVX512ER(خالی) { واپس f_7_ebx[27]; }
    منطقی AVX512CD(خالی) { واپس f_7_ebx[28]; }
    منطقی AVX512BW(خالی) { واپس f_7_ebx[30]; }
    منطقی AVX512VL(خالی) { واپس f_7_ebx[31]; }

    منطقی SHA(خالی) { واپس f_7_ebx[29]; }

    منطقی PREFETCHWT1(خالی) { واپس f_7_ecx[0]; }

    منطقی LAHF(خالی) { واپس f_81_ecx[0]; }
    منطقی LZCNT(خالی) { واپس is_intel && f_81_ecx[5]; }
    منطقی ABM(خالی) { واپس is_amd && f_81_ecx[5]; }
    منطقی SSE4a(خالی) { واپس is_amd && f_81_ecx[6]; }
    منطقی XOP(خالی) { واپس is_amd && f_81_ecx[11]; }
    منطقی TBM(خالی) { واپس is_amd && f_81_ecx[21]; }

    منطقی SYSCALL(خالی) { واپس is_intel && f_81_edx[11]; }
    منطقی MMXEXT(خالی) { واپس is_amd && f_81_edx[22]; }
    منطقی RDTSCP(خالی) { واپس is_intel && f_81_edx[27]; }
    منطقی _3DNOWEXT(خالی) { واپس is_amd && f_81_edx[30]; }
    منطقی _3DNOW(خالی) { واپس is_amd && f_81_edx[31]; }

    منطقی AVX512_VBMI(خالی) { واپس f_7_ecx[1]; }
    منطقی AVX512_VNNI(خالی) { واپس f_7_ecx[11]; }
    منطقی AVX512_FP16(خالی) { واپس f_7_edx[23]; }
    منطقی AVX512_BF16(خالی) { واپس f_7_1_eax[5]; }
    منطقی AVX_VNNI(خالی) { واپس f_7_1_eax[4]; }

    منطقی AMX_TILE(خالی) { واپس f_7_edx[24]; }
    منطقی AMX_INT8(خالی) { واپس f_7_edx[25]; }
    منطقی AMX_FP16(خالی) { واپس f_7_1_eax[21]; }
    منطقی AMX_BF16(خالی) { واپس f_7_edx[22]; }

#اگر_تعریف_شدہ _MSC_VER
    ساکن خالی cpuid(صحیح cpu_info[4], صحیح eax) {
        __cpuid(cpu_info, eax);
    }
    ساکن خالی cpuidex(صحیح cpu_info[4], صحیح eax, صحیح ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ورنہ
    ساکن خالی cpuid(صحیح cpu_info[4], صحیح eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    ساکن خالی cpuidex(صحیح cpu_info[4], صحیح eax, صحیح ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#اگر_ختم

    cpuid_x86() {
        std::array<صحیح, 4> cpui;
        std::vector<std::array<صحیح, 4>> ڈیٹا;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.ڈیٹا(), 0);
        صحیح n_ids = cpui[0];

        برائے (صحیح i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.ڈیٹا(), i, 0);
            ڈیٹا.push_back(cpui);
        }

        // capture vendor string
        حرف vendor[0x20] = {};
        *reinterpret_cast<صحیح *>(vendor)     = ڈیٹا[0][1];
        *reinterpret_cast<صحیح *>(vendor + 4) = ڈیٹا[0][3];
        *reinterpret_cast<صحیح *>(vendor + 8) = ڈیٹا[0][2];
        یہ->vendor = vendor;
        اگر (یہ->vendor == "GenuineIntel") {
            is_intel = سچ;
        } ورنہ اگر (یہ->vendor == "AuthenticAMD") {
            is_amd = سچ;
        }

        // load bitset with flags for function 0x00000001
        اگر (n_ids >= 1) {
            f_1_ecx = ڈیٹا[1][2];
            f_1_edx = ڈیٹا[1][3];
        }

        // load bitset with flags for function 0x00000007
        اگر (n_ids >= 7) {
            f_7_ebx = ڈیٹا[7][1];
            f_7_ecx = ڈیٹا[7][2];
            f_7_edx = ڈیٹا[7][3];
            cpuidex(cpui.ڈیٹا(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.ڈیٹا(), 0x80000000);
        غیرعلامتی صحیح n_ex_ids = cpui[0];

        std::vector<std::array<صحیح, 4>> ext_ڈیٹا;
        برائے (غیرعلامتی صحیح i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.ڈیٹا(), i, 0);
            ext_ڈیٹا.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        اگر (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_ڈیٹا[1][2];
            f_81_edx = ext_ڈیٹا[1][3];
        }

        // interpret CPU brand string if reported
        حرف brand[0x40] = {};
        اگر (n_ex_ids >= 0x80000004) {
            std::حافظہ_نقل(brand, ext_ڈیٹا[2].ڈیٹا(), حجم(cpui));
            std::حافظہ_نقل(brand + 16, ext_ڈیٹا[3].ڈیٹا(), حجم(cpui));
            std::حافظہ_نقل(brand + 32, ext_ڈیٹا[4].ڈیٹا(), حجم(cpui));
            یہ->brand = brand;
        }
    }

    منطقی is_intel = جھوٹ;
    منطقی is_amd = جھوٹ;
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
خالی test_x86_is() {
    cpuid_x86 is;
    لکھو("CPU Vendor: %s\n", is.vendor.c_str());
    لکھو("Brand: %s\n", is.brand.c_str());
    لکھو("is_intel: %d\n", is.is_intel);
    لکھو("is_amd: %d\n", is.is_amd);
    لکھو("sse3: %d\n", is.SSE3());
    لکھو("pclmulqdq: %d\n", is.PCLMULQDQ());
    لکھو("ssse3: %d\n", is.SSSE3());
    لکھو("fma: %d\n", is.FMA());
    لکھو("cmpxchg16b: %d\n", is.CMPXCHG16B());
    لکھو("sse41: %d\n", is.SSE41());
    لکھو("sse42: %d\n", is.SSE42());
    لکھو("movbe: %d\n", is.MOVBE());
    لکھو("popcnt: %d\n", is.POPCNT());
    لکھو("aes: %d\n", is.AES());
    لکھو("xsave: %d\n", is.XSAVE());
    لکھو("osxsave: %d\n", is.OSXSAVE());
    لکھو("avx: %d\n", is.AVX());
    لکھو("f16c: %d\n", is.F16C());
    لکھو("rdrand: %d\n", is.RDRAND());
    لکھو("msr: %d\n", is.MSR());
    لکھو("cx8: %d\n", is.CX8());
    لکھو("sep: %d\n", is.SEP());
    لکھو("cmov: %d\n", is.CMOV());
    لکھو("clflush: %d\n", is.CLFSH());
    لکھو("mmx: %d\n", is.MMX());
    لکھو("fxsr: %d\n", is.FXSR());
    لکھو("sse: %d\n", is.SSE());
    لکھو("sse2: %d\n", is.SSE2());
    لکھو("fsgsbase: %d\n", is.FSGSBASE());
    لکھو("bmi1: %d\n", is.BMI1());
    لکھو("hle: %d\n", is.HLE());
    لکھو("avx2: %d\n", is.AVX2());
    لکھو("bmi2: %d\n", is.BMI2());
    لکھو("erms: %d\n", is.ERMS());
    لکھو("invpcid: %d\n", is.INVPCID());
    لکھو("rtm: %d\n", is.RTM());
    لکھو("avx512f: %d\n", is.AVX512F());
    لکھو("rdseed: %d\n", is.RDSEED());
    لکھو("adx: %d\n", is.ADX());
    لکھو("avx512pf: %d\n", is.AVX512PF());
    لکھو("avx512er: %d\n", is.AVX512ER());
    لکھو("avx512cd: %d\n", is.AVX512CD());
    لکھو("sha: %d\n", is.SHA());
    لکھو("prefetchwt1: %d\n", is.PREFETCHWT1());
    لکھو("lahf: %d\n", is.LAHF());
    لکھو("lzcnt: %d\n", is.LZCNT());
    لکھو("abm: %d\n", is.ABM());
    لکھو("sse4a: %d\n", is.SSE4a());
    لکھو("xop: %d\n", is.XOP());
    لکھو("tbm: %d\n", is.TBM());
    لکھو("syscall: %d\n", is.SYSCALL());
    لکھو("mmxext: %d\n", is.MMXEXT());
    لکھو("rdtscp: %d\n", is.RDTSCP());
    لکھو("3dnowext: %d\n", is._3DNOWEXT());
    لکھو("3dnow: %d\n", is._3DNOW());
    لکھو("avx512_vbmi: %d\n", is.AVX512_VBMI());
    لکھو("avx512_vnni: %d\n", is.AVX512_VNNI());
    لکھو("avx512_fp16: %d\n", is.AVX512_FP16());
    لکھو("avx512_bf16: %d\n", is.AVX512_BF16());
    لکھو("amx_tile: %d\n", is.AMX_TILE());
    لکھو("amx_int8: %d\n", is.AMX_INT8());
    لکھو("amx_fp16: %d\n", is.AMX_FP16());
    لکھو("amx_bf16: %d\n", is.AMX_BF16());
}
#اگر_ختم

ساکن صحیح ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    صحیح score = 1;
    cpuid_x86 is;

#اگر_تعریف_شدہ GGML_FMA
    اگر (!is.FMA()) { واپس 0; }
    score += 1;
#اگر_ختم
#اگر_تعریف_شدہ GGML_F16C
    اگر (!is.F16C()) { واپس 0; }
    score += 1<<1;
#اگر_ختم
#اگر_تعریف_شدہ GGML_SSE42
    اگر (!is.SSE42()) { واپس 0; }
    score += 1<<2;
#اگر_ختم
#اگر_تعریف_شدہ GGML_BMI2
    اگر (!is.BMI2()) { واپس 0; }
    score += 1<<3;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX
    اگر (!is.AVX()) { واپس 0; }
    score += 1<<4;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX2
    اگر (!is.AVX2()) { واپس 0; }
    score += 1<<5;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX_VNNI
    اگر (!is.AVX_VNNI()) { واپس 0; }
    score += 1<<6;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX512
    اگر (!is.AVX512F()) { واپس 0; }
    اگر (!is.AVX512CD()) { واپس 0; }
    اگر (!is.AVX512VL()) { واپس 0; }
    اگر (!is.AVX512DQ()) { واپس 0; }
    اگر (!is.AVX512BW()) { واپس 0; }
    score += 1<<7;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX512_VBMI
    اگر (!is.AVX512_VBMI()) { واپس 0; }
    score += 1<<8;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX512_BF16
    اگر (!is.AVX512_BF16()) { واپس 0; }
    score += 1<<9;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AVX512_VNNI
    اگر (!is.AVX512_VNNI()) { واپس 0; }
    score += 1<<10;
#اگر_ختم
#اگر_تعریف_شدہ GGML_AMX_INT8
    اگر (!is.AMX_INT8()) { واپس 0; }
    score += 1<<11;
#اگر_ختم

    واپس score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#اگر_ختم // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
