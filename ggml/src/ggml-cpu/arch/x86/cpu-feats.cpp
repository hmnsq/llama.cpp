#include "ggml-backend-impl.h"

#ຖ້າ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstring>
#include <vector>
#include <bitset>
#include <array>
#include <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
sa_nracanaa cpuid_x86 {
    bool SSE3(void) { ສົ່ງຄືນ f_1_ecx[0]; }
    bool PCLMULQDQ(void) { ສົ່ງຄືນ f_1_ecx[1]; }
    bool MONITOR(void) { ສົ່ງຄືນ f_1_ecx[3]; }
    bool SSSE3(void) { ສົ່ງຄືນ f_1_ecx[9]; }
    bool FMA(void) { ສົ່ງຄືນ f_1_ecx[12]; }
    bool CMPXCHG16B(void) { ສົ່ງຄືນ f_1_ecx[13]; }
    bool SSE41(void) { ສົ່ງຄືນ f_1_ecx[19]; }
    bool SSE42(void) { ສົ່ງຄືນ f_1_ecx[20]; }
    bool MOVBE(void) { ສົ່ງຄືນ f_1_ecx[22]; }
    bool POPCNT(void) { ສົ່ງຄືນ f_1_ecx[23]; }
    bool AES(void) { ສົ່ງຄືນ f_1_ecx[25]; }
    bool XSAVE(void) { ສົ່ງຄືນ f_1_ecx[26]; }
    bool OSXSAVE(void) { ສົ່ງຄືນ f_1_ecx[27]; }
    bool AVX(void) { ສົ່ງຄືນ f_1_ecx[28]; }
    bool F16C(void) { ສົ່ງຄືນ f_1_ecx[29]; }
    bool RDRAND(void) { ສົ່ງຄືນ f_1_ecx[30]; }

    bool MSR(void) { ສົ່ງຄືນ f_1_edx[5]; }
    bool CX8(void) { ສົ່ງຄືນ f_1_edx[8]; }
    bool SEP(void) { ສົ່ງຄືນ f_1_edx[11]; }
    bool CMOV(void) { ສົ່ງຄືນ f_1_edx[15]; }
    bool CLFSH(void) { ສົ່ງຄືນ f_1_edx[19]; }
    bool MMX(void) { ສົ່ງຄືນ f_1_edx[23]; }
    bool FXSR(void) { ສົ່ງຄືນ f_1_edx[24]; }
    bool SSE(void) { ສົ່ງຄືນ f_1_edx[25]; }
    bool SSE2(void) { ສົ່ງຄືນ f_1_edx[26]; }

    bool FSGSBASE(void) { ສົ່ງຄືນ f_7_ebx[0]; }
    bool BMI1(void) { ສົ່ງຄືນ f_7_ebx[3]; }
    bool HLE(void) { ສົ່ງຄືນ is_intel && f_7_ebx[4]; }
    bool AVX2(void) { ສົ່ງຄືນ f_7_ebx[5]; }
    bool BMI2(void) { ສົ່ງຄືນ f_7_ebx[8]; }
    bool ERMS(void) { ສົ່ງຄືນ f_7_ebx[9]; }
    bool INVPCID(void) { ສົ່ງຄືນ f_7_ebx[10]; }
    bool RTM(void) { ສົ່ງຄືນ is_intel && f_7_ebx[11]; }
    bool AVX512F(void) { ສົ່ງຄືນ f_7_ebx[16]; }
    bool AVX512DQ(void) { ສົ່ງຄືນ f_7_ebx[17]; }
    bool RDSEED(void) { ສົ່ງຄືນ f_7_ebx[18]; }
    bool ADX(void) { ສົ່ງຄືນ f_7_ebx[19]; }
    bool AVX512PF(void) { ສົ່ງຄືນ f_7_ebx[26]; }
    bool AVX512ER(void) { ສົ່ງຄືນ f_7_ebx[27]; }
    bool AVX512CD(void) { ສົ່ງຄືນ f_7_ebx[28]; }
    bool AVX512BW(void) { ສົ່ງຄືນ f_7_ebx[30]; }
    bool AVX512VL(void) { ສົ່ງຄືນ f_7_ebx[31]; }

    bool SHA(void) { ສົ່ງຄືນ f_7_ebx[29]; }

    bool PREFETCHWT1(void) { ສົ່ງຄືນ f_7_ecx[0]; }

    bool LAHF(void) { ສົ່ງຄືນ f_81_ecx[0]; }
    bool LZCNT(void) { ສົ່ງຄືນ is_intel && f_81_ecx[5]; }
    bool ABM(void) { ສົ່ງຄືນ is_amd && f_81_ecx[5]; }
    bool SSE4a(void) { ສົ່ງຄືນ is_amd && f_81_ecx[6]; }
    bool XOP(void) { ສົ່ງຄືນ is_amd && f_81_ecx[11]; }
    bool TBM(void) { ສົ່ງຄືນ is_amd && f_81_ecx[21]; }

    bool SYSCALL(void) { ສົ່ງຄືນ is_intel && f_81_edx[11]; }
    bool MMXEXT(void) { ສົ່ງຄືນ is_amd && f_81_edx[22]; }
    bool RDTSCP(void) { ສົ່ງຄືນ is_intel && f_81_edx[27]; }
    bool _3DNOWEXT(void) { ສົ່ງຄືນ is_amd && f_81_edx[30]; }
    bool _3DNOW(void) { ສົ່ງຄືນ is_amd && f_81_edx[31]; }

    bool AVX512_VBMI(void) { ສົ່ງຄືນ f_7_ecx[1]; }
    bool AVX512_VNNI(void) { ສົ່ງຄືນ f_7_ecx[11]; }
    bool AVX512_FP16(void) { ສົ່ງຄືນ f_7_edx[23]; }
    bool AVX512_BF16(void) { ສົ່ງຄືນ f_7_1_eax[5]; }
    bool AVX_VNNI(void) { ສົ່ງຄືນ f_7_1_eax[4]; }

    bool AMX_TILE(void) { ສົ່ງຄືນ f_7_edx[24]; }
    bool AMX_INT8(void) { ສົ່ງຄືນ f_7_edx[25]; }
    bool AMX_FP16(void) { ສົ່ງຄືນ f_7_1_eax[21]; }
    bool AMX_BF16(void) { ສົ່ງຄືນ f_7_edx[22]; }

#ifdef _MSC_VER
    static void cpuid(ຈຳນວນເຕັມ cpu_info[4], ຈຳນວນເຕັມ eax) {
        __cpuid(cpu_info, eax);
    }
    static void cpuidex(ຈຳນວນເຕັມ cpu_info[4], ຈຳນວນເຕັມ eax, ຈຳນວນເຕັມ ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#ບໍ່ດັ່ງນັ້ນ
    static void cpuid(ຈຳນວນເຕັມ cpu_info[4], ຈຳນວນເຕັມ eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    static void cpuidex(ຈຳນວນເຕັມ cpu_info[4], ຈຳນວນເຕັມ eax, ຈຳນວນເຕັມ ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<ຈຳນວນເຕັມ, 4> cpui;
        std::vector<std::array<ຈຳນວນເຕັມ, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        ຈຳນວນເຕັມ n_ids = cpui[0];

        ສຳລັບ (ຈຳນວນເຕັມ i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        var_na vendor[0x20] = {};
        *reinterpret_cast<ຈຳນວນເຕັມ *>(vendor)     = data[0][1];
        *reinterpret_cast<ຈຳນວນເຕັມ *>(vendor + 4) = data[0][3];
        *reinterpret_cast<ຈຳນວນເຕັມ *>(vendor + 8) = data[0][2];
        this->vendor = vendor;
        ຖ້າ (this->vendor == "GenuineIntel") {
            is_intel = true;
        } ບໍ່ດັ່ງນັ້ນ ຖ້າ (this->vendor == "AuthenticAMD") {
            is_amd = true;
        }

        // load bitset with flags for function 0x00000001
        ຖ້າ (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        ຖ້າ (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned ຈຳນວນເຕັມ n_ex_ids = cpui[0];

        std::vector<std::array<ຈຳນວນເຕັມ, 4>> ext_data;
        ສຳລັບ (unsigned ຈຳນວນເຕັມ i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        ຖ້າ (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        var_na brand[0x40] = {};
        ຖ້າ (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), parimaa_na(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), parimaa_na(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), parimaa_na(cpui));
            this->brand = brand;
        }
    }

    bool is_intel = false;
    bool is_amd = false;
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

#ຖ້າ 0
void test_x86_is() {
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

static ຈຳນວນເຕັມ ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    ຈຳນວນເຕັມ score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    ຖ້າ (!is.FMA()) { ສົ່ງຄືນ 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    ຖ້າ (!is.F16C()) { ສົ່ງຄືນ 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    ຖ້າ (!is.SSE42()) { ສົ່ງຄືນ 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    ຖ້າ (!is.BMI2()) { ສົ່ງຄືນ 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    ຖ້າ (!is.AVX()) { ສົ່ງຄືນ 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    ຖ້າ (!is.AVX2()) { ສົ່ງຄືນ 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    ຖ້າ (!is.AVX_VNNI()) { ສົ່ງຄືນ 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    ຖ້າ (!is.AVX512F()) { ສົ່ງຄືນ 0; }
    ຖ້າ (!is.AVX512CD()) { ສົ່ງຄືນ 0; }
    ຖ້າ (!is.AVX512VL()) { ສົ່ງຄືນ 0; }
    ຖ້າ (!is.AVX512DQ()) { ສົ່ງຄືນ 0; }
    ຖ້າ (!is.AVX512BW()) { ສົ່ງຄືນ 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    ຖ້າ (!is.AVX512_VBMI()) { ສົ່ງຄືນ 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    ຖ້າ (!is.AVX512_BF16()) { ສົ່ງຄືນ 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    ຖ້າ (!is.AVX512_VNNI()) { ສົ່ງຄືນ 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    ຖ້າ (!is.AMX_INT8()) { ສົ່ງຄືນ 0; }
    score += 1<<11;
#endif

    ສົ່ງຄືນ score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
