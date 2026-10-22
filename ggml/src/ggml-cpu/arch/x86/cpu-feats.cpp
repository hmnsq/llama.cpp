#include "ggml-backend-impl.h"

#만약 defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstring>
#include <vector>
#include <bitset>
#include <array>
#include <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
struct cpuid_x86 {
    bool SSE3(void) { 반환 f_1_ecx[0]; }
    bool PCLMULQDQ(void) { 반환 f_1_ecx[1]; }
    bool MONITOR(void) { 반환 f_1_ecx[3]; }
    bool SSSE3(void) { 반환 f_1_ecx[9]; }
    bool FMA(void) { 반환 f_1_ecx[12]; }
    bool CMPXCHG16B(void) { 반환 f_1_ecx[13]; }
    bool SSE41(void) { 반환 f_1_ecx[19]; }
    bool SSE42(void) { 반환 f_1_ecx[20]; }
    bool MOVBE(void) { 반환 f_1_ecx[22]; }
    bool POPCNT(void) { 반환 f_1_ecx[23]; }
    bool AES(void) { 반환 f_1_ecx[25]; }
    bool XSAVE(void) { 반환 f_1_ecx[26]; }
    bool OSXSAVE(void) { 반환 f_1_ecx[27]; }
    bool AVX(void) { 반환 f_1_ecx[28]; }
    bool F16C(void) { 반환 f_1_ecx[29]; }
    bool RDRAND(void) { 반환 f_1_ecx[30]; }

    bool MSR(void) { 반환 f_1_edx[5]; }
    bool CX8(void) { 반환 f_1_edx[8]; }
    bool SEP(void) { 반환 f_1_edx[11]; }
    bool CMOV(void) { 반환 f_1_edx[15]; }
    bool CLFSH(void) { 반환 f_1_edx[19]; }
    bool MMX(void) { 반환 f_1_edx[23]; }
    bool FXSR(void) { 반환 f_1_edx[24]; }
    bool SSE(void) { 반환 f_1_edx[25]; }
    bool SSE2(void) { 반환 f_1_edx[26]; }

    bool FSGSBASE(void) { 반환 f_7_ebx[0]; }
    bool BMI1(void) { 반환 f_7_ebx[3]; }
    bool HLE(void) { 반환 is_intel && f_7_ebx[4]; }
    bool AVX2(void) { 반환 f_7_ebx[5]; }
    bool BMI2(void) { 반환 f_7_ebx[8]; }
    bool ERMS(void) { 반환 f_7_ebx[9]; }
    bool INVPCID(void) { 반환 f_7_ebx[10]; }
    bool RTM(void) { 반환 is_intel && f_7_ebx[11]; }
    bool AVX512F(void) { 반환 f_7_ebx[16]; }
    bool AVX512DQ(void) { 반환 f_7_ebx[17]; }
    bool RDSEED(void) { 반환 f_7_ebx[18]; }
    bool ADX(void) { 반환 f_7_ebx[19]; }
    bool AVX512PF(void) { 반환 f_7_ebx[26]; }
    bool AVX512ER(void) { 반환 f_7_ebx[27]; }
    bool AVX512CD(void) { 반환 f_7_ebx[28]; }
    bool AVX512BW(void) { 반환 f_7_ebx[30]; }
    bool AVX512VL(void) { 반환 f_7_ebx[31]; }

    bool SHA(void) { 반환 f_7_ebx[29]; }

    bool PREFETCHWT1(void) { 반환 f_7_ecx[0]; }

    bool LAHF(void) { 반환 f_81_ecx[0]; }
    bool LZCNT(void) { 반환 is_intel && f_81_ecx[5]; }
    bool ABM(void) { 반환 is_amd && f_81_ecx[5]; }
    bool SSE4a(void) { 반환 is_amd && f_81_ecx[6]; }
    bool XOP(void) { 반환 is_amd && f_81_ecx[11]; }
    bool TBM(void) { 반환 is_amd && f_81_ecx[21]; }

    bool SYSCALL(void) { 반환 is_intel && f_81_edx[11]; }
    bool MMXEXT(void) { 반환 is_amd && f_81_edx[22]; }
    bool RDTSCP(void) { 반환 is_intel && f_81_edx[27]; }
    bool _3DNOWEXT(void) { 반환 is_amd && f_81_edx[30]; }
    bool _3DNOW(void) { 반환 is_amd && f_81_edx[31]; }

    bool AVX512_VBMI(void) { 반환 f_7_ecx[1]; }
    bool AVX512_VNNI(void) { 반환 f_7_ecx[11]; }
    bool AVX512_FP16(void) { 반환 f_7_edx[23]; }
    bool AVX512_BF16(void) { 반환 f_7_1_eax[5]; }
    bool AVX_VNNI(void) { 반환 f_7_1_eax[4]; }

    bool AMX_TILE(void) { 반환 f_7_edx[24]; }
    bool AMX_INT8(void) { 반환 f_7_edx[25]; }
    bool AMX_FP16(void) { 반환 f_7_1_eax[21]; }
    bool AMX_BF16(void) { 반환 f_7_edx[22]; }

#ifdef _MSC_VER
    static void cpuid(정수 cpu_info[4], 정수 eax) {
        __cpuid(cpu_info, eax);
    }
    static void cpuidex(정수 cpu_info[4], 정수 eax, 정수 ecx) {
        __cpuidex(cpu_info, eax, ecx);
    }
#아니면
    static void cpuid(정수 cpu_info[4], 정수 eax) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(0));
    }
    static void cpuidex(정수 cpu_info[4], 정수 eax, 정수 ecx) {
        __asm__ __volatile__(
            "cpuid"
            : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
            : "a"(eax), "c"(ecx));
    }
#endif

    cpuid_x86() {
        std::array<정수, 4> cpui;
        std::vector<std::array<정수, 4>> data;

        // calling __cpuid with 0x0 as the function_id argument
        // gets the number of the highest valid function ID.
        cpuid(cpui.data(), 0);
        정수 n_ids = cpui[0];

        위해 (정수 i = 0; i <= n_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
        }

        // capture vendor string
        char vendor[0x20] = {};
        *reinterpret_cast<정수 *>(vendor)     = data[0][1];
        *reinterpret_cast<정수 *>(vendor + 4) = data[0][3];
        *reinterpret_cast<정수 *>(vendor + 8) = data[0][2];
        this->vendor = vendor;
        만약 (this->vendor == "GenuineIntel") {
            is_intel = true;
        } 아니면 만약 (this->vendor == "AuthenticAMD") {
            is_amd = true;
        }

        // load bitset with flags for function 0x00000001
        만약 (n_ids >= 1) {
            f_1_ecx = data[1][2];
            f_1_edx = data[1][3];
        }

        // load bitset with flags for function 0x00000007
        만약 (n_ids >= 7) {
            f_7_ebx = data[7][1];
            f_7_ecx = data[7][2];
            f_7_edx = data[7][3];
            cpuidex(cpui.data(), 7, 1);
            f_7_1_eax = cpui[0];
        }

        // calling __cpuid with 0x80000000 as the function_id argument
        // gets the number of the highest valid extended ID.
        cpuid(cpui.data(), 0x80000000);
        unsigned 정수 n_ex_ids = cpui[0];

        std::vector<std::array<정수, 4>> ext_data;
        위해 (unsigned 정수 i = 0x80000000; i <= n_ex_ids; ++i) {
            cpuidex(cpui.data(), i, 0);
            ext_data.push_back(cpui);
        }

        // load bitset with flags for function 0x80000001
        만약 (n_ex_ids >= 0x80000001) {
            f_81_ecx = ext_data[1][2];
            f_81_edx = ext_data[1][3];
        }

        // interpret CPU brand string if reported
        char brand[0x40] = {};
        만약 (n_ex_ids >= 0x80000004) {
            std::memcpy(brand, ext_data[2].data(), sizeof(cpui));
            std::memcpy(brand + 16, ext_data[3].data(), sizeof(cpui));
            std::memcpy(brand + 32, ext_data[4].data(), sizeof(cpui));
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

#만약 0
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

static 정수 ggml_backend_cpu_x86_score() {
    // FIXME: this does not check for OS support

    정수 score = 1;
    cpuid_x86 is;

#ifdef GGML_FMA
    만약 (!is.FMA()) { 반환 0; }
    score += 1;
#endif
#ifdef GGML_F16C
    만약 (!is.F16C()) { 반환 0; }
    score += 1<<1;
#endif
#ifdef GGML_SSE42
    만약 (!is.SSE42()) { 반환 0; }
    score += 1<<2;
#endif
#ifdef GGML_BMI2
    만약 (!is.BMI2()) { 반환 0; }
    score += 1<<3;
#endif
#ifdef GGML_AVX
    만약 (!is.AVX()) { 반환 0; }
    score += 1<<4;
#endif
#ifdef GGML_AVX2
    만약 (!is.AVX2()) { 반환 0; }
    score += 1<<5;
#endif
#ifdef GGML_AVX_VNNI
    만약 (!is.AVX_VNNI()) { 반환 0; }
    score += 1<<6;
#endif
#ifdef GGML_AVX512
    만약 (!is.AVX512F()) { 반환 0; }
    만약 (!is.AVX512CD()) { 반환 0; }
    만약 (!is.AVX512VL()) { 반환 0; }
    만약 (!is.AVX512DQ()) { 반환 0; }
    만약 (!is.AVX512BW()) { 반환 0; }
    score += 1<<7;
#endif
#ifdef GGML_AVX512_VBMI
    만약 (!is.AVX512_VBMI()) { 반환 0; }
    score += 1<<8;
#endif
#ifdef GGML_AVX512_BF16
    만약 (!is.AVX512_BF16()) { 반환 0; }
    score += 1<<9;
#endif
#ifdef GGML_AVX512_VNNI
    만약 (!is.AVX512_VNNI()) { 반환 0; }
    score += 1<<10;
#endif
#ifdef GGML_AMX_INT8
    만약 (!is.AMX_INT8()) { 반환 0; }
    score += 1<<11;
#endif

    반환 score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_x86_score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
