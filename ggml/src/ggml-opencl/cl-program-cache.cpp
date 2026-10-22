// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#ವ್ಯಾಖ್ಯಾನಿಸು CL_ಗುರಿ_OPENCL_ಆವೃತ್ತಿ GGML_OPENCL_ಗುರಿ_ಆವೃತ್ತಿ
#ವ್ಯಾಖ್ಯಾನಿಸು CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ಅನ್ತರ್ಭಾವ "cl-program-cache.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"  // GGML_LOG_INFO / WARN

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <filesystem>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <system_error>
#ಅನ್ತರ್ಭಾವ <vector>

#ಆದರೆ defined(_WIN32)
#  ifndef WIN32_LEAN_AND_ಸರಾಸರಿ
#    define WIN32_LEAN_AND_ಸರಾಸರಿ
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((ಪೂರ್ಣಾಂಕ) GetCurrentProcessId())
#ಇಲ್ಲದಿದ್ದರೆ
#  include <unistd.h>
#  define ggml_getpid()    ((ಪೂರ್ಣಾಂಕ) getpid())
#ಆದರೆ_ಅಂತ್ಯ

ನಾಮಾಕಾಶ fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

ನಾಮಾಕಾಶ {

ರಚನೆ sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buf_len;
};

ಸ್ಥಿರಾಂಕ uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

ಸಾಲೊಳಗೆ uint32_t rotr32(uint32_t x, ಚಿಹ್ನೆರಹಿತ n) { ಹಿಂತಿರುಗಿ (x >> n) | (x << (32 - n)); }

ಶೂನ್ಯ sha256_compress(uint32_t state[8], ಸ್ಥಿರಾಂಕ uint8_t ಖಂಡ[64]) {
    uint32_t w[64];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)ಖಂಡ[i*4    ] << 24) |
               ((uint32_t)ಖಂಡ[i*4 + 1] << 16) |
               ((uint32_t)ಖಂಡ[i*4 + 2] <<  8) |
               ((uint32_t)ಖಂಡ[i*4 + 3]      );
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 64; ++i) {
        uint32_t S1   = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch   = (e & f) ^ ((~e) & g);
        uint32_t t1   = h + S1 + ch + K256[i] + w[i];
        uint32_t S0   = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj  = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2   = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

ಶೂನ್ಯ sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

ಶೂನ್ಯ sha256_update(sha256_ctx & c, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t len) {
    ಸ್ಥಿರಾಂಕ uint8_t * p = (ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ;
    c.bitlen += (uint64_t) len * 8;
    ಆದರೆ (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        ಆದರೆ (n > len) { n = len; }
        ಸ್ಮರಣೆ_ನಕಲು(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        ಆದರೆ (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    ಆಗ (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    ಆದರೆ (len > 0) {
        ಸ್ಮರಣೆ_ನಕಲು(c.buf, p, len);
        c.buf_len = len;
    }
}

ಶೂನ್ಯ sha256_final(sha256_ctx & c, uint8_t ಹೊರಹೋಗು[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    ಆದರೆ (c.buf_len > 56) {
        ಆಗ (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    ಆಗ (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
        ಹೊರಹೋಗು[i*4    ] = (uint8_t) (c.state[i] >> 24);
        ಹೊರಹೋಗು[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        ಹೊರಹೋಗು[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        ಹೊರಹೋಗು[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(ಸ್ಥಿರಾಂಕ uint8_t digest[32]) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ hex[] = "0123456789abcdef";
    std::string s(64, '0');
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    ಹಿಂತಿರುಗಿ s;
}

std::string compute_ಕೀಲಿ(ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ_suffix,
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *        ಮೂಲ,
                        ಸ್ಥಿರಾಂಕ std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t sep = 0;
    sha256_update(c, ಮೂಲ,             ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಮೂಲ));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.ದತ್ತಾಂಶ(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, ಕೀಲಿ_suffix.ದತ್ತಾಂಶ(),  ಕೀಲಿ_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    ಹಿಂತಿರುಗಿ sha256_hex(digest);
}

ತಾರ್ಕಿಕ make_dir_recursive(ಸ್ಥಿರಾಂಕ std::string & path) {
    ಆದರೆ (path.ಖಾಲಿಗಳು()) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    ಸ್ಥಿರಾಂಕ fs::path p = fs::u8path(path);
    std::error_code ec;
    ಆದರೆ (fs::create_directories(p, ec)) { ಹಿಂತಿರುಗಿ ನಿಜ; }
    std::error_code ec_stat;
    ಹಿಂತಿರುಗಿ fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#ಆದರೆ defined(_WIN32)
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = std::getenv("LOCALAPPDATA");
    ಆದರೆ (!ತಳ || !*ತಳ) { ತಳ = std::getenv("APPDATA"); }
    ಆದರೆ (!ತಳ || !*ತಳ) { ತಳ = std::getenv("TEMP"); }
    ಆದರೆ (!ತಳ || !*ತಳ) { ತಳ = "."; }
    ಹಿಂತಿರುಗಿ std::string(ತಳ) + "\\llama.cpp\\cl-cache";
#ಅಥವಾಆದರೆ defined(__APPLE__)
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * home = std::getenv("HOME");
    ಆದರೆ (!home || !*home) { home = "."; }
    ಹಿಂತಿರುಗಿ std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#ಇಲ್ಲದಿದ್ದರೆ
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    ಸ್ಥಿರಾಂಕ fs::path tmp_ಮಾರ್ಗ = fs::temp_directory_path(ec);
    ಆದರೆ (ec || tmp_ಮಾರ್ಗ.ಖಾಲಿಗಳು()) { ಹಿಂತಿರುಗಿ {}; }
    ಹಿಂತಿರುಗಿ tmp_ಮಾರ್ಗ.string() + "/llama.cpp/cl-cache";
#ಆದರೆ_ಅಂತ್ಯ
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು GetInfoFn, ಬಗೆ_ಹೆಸರು Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    ಆದರೆ (fn(obj, name, 0, ಶೂನ್ಯಸೂಚಕ, &sz) != CL_SUCCESS || sz == 0) {
        ಹಿಂತಿರುಗಿ {};
    }
    std::string s(sz, '\0');
    ಆದರೆ (fn(obj, name, sz, &s[0], ಶೂನ್ಯಸೂಚಕ) != CL_SUCCESS) {
        ಹಿಂತಿರುಗಿ {};
    }
    ಆದರೆ (!s.ಖಾಲಿಗಳು() && s.back() == '\0') {
        s.pop_back();
    }
    ಹಿಂತಿರುಗಿ s;
}

std::string compute_ಕೀಲಿ_suffix(cl_device_id device) {
    cl_platform_id platform = ಶೂನ್ಯಸೂಚಕ;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, ಗಾತ್ರ(platform), &platform, ಶೂನ್ಯಸೂಚಕ);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_ಆವೃತ್ತಿ);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_ಆವೃತ್ತಿ);    s.push_back('\0');
    ಆದರೆ (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_ಆವೃತ್ತಿ); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_ಕಾರ್ಯಕ್ರಮ_CACHE_FORMAT_ಆವೃತ್ತಿ);
    ಹಿಂತಿರುಗಿ s;
}

ಸ್ಥಿರಾಂಕ uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

ತಾರ್ಕಿಕ read_all(ಸ್ಥಿರಾಂಕ std::string & path, std::vector<uint8_t> & ಹೊರಹೋಗು) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    ಆದರೆ (!f) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    ಆದರೆ (sz < 0) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    f.seekg(0, std::ios::beg);
    ಹೊರಹೋಗು.resize((size_t) sz);
    ಆದರೆ (sz > 0) { f.ಪಠ((ಅಕ್ಷರ *) ಹೊರಹೋಗು.ದತ್ತಾಂಶ(), sz); }
    ಹಿಂತಿರುಗಿ f.good() || f.eof();
}

ತಾರ್ಕಿಕ write_atomic(ಸ್ಥಿರಾಂಕ std::string & path, ಸ್ಥಿರಾಂಕ uint8_t * ದತ್ತಾಂಶ, size_t len) {
    ಸ್ಥಿರಾಂಕ fs::path dst = fs::u8path(path);
    ಸ್ಥಿರಾಂಕ fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        ಆದರೆ (!f) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
        f.ಲಿಖ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ, (std::streamsize) len);
        ಆದರೆ (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    ಆದರೆ (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

}  // namespace

ಸ್ಥಿರ ತಾರ್ಕಿಕ cache_debug_enabled() {
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ cached = -1;
    ಆದರೆ (cached < 0) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    ಹಿಂತಿರುಗಿ cached != 0;
}

ಸ್ಥಿರ std::string opts_preview(ಸ್ಥಿರಾಂಕ std::string & opts, size_t n = 120) {
    ಆದರೆ (opts.size() <= n) { ಹಿಂತಿರುಗಿ opts; }
    ಹಿಂತಿರುಗಿ opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ g_cache_ದೊರಕಿದವು = 0, g_cache_ತಪ್ಪಿದವು = 0, g_cache_saves = 0;

// Debug trace directly to stderr
ಸ್ಥಿರ ಶೂನ್ಯ cache_debug_ಸಾಲು(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಬಗೆ, ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ,
                             ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮೂಲ, ಸ್ಥಿರಾಂಕ std::string & opts) {
    ಆದರೆ (!cache_debug_enabled()) { ಹಿಂತಿರುಗಿ; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            ಬಗೆ, g_cache_ದೊರಕಿದವು, g_cache_ತಪ್ಪಿದವು, g_cache_saves,
            ಕೀಲಿ.substr(0, 16).c_str(), ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಮೂಲ), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_ಕಾರ್ಯಕ್ರಮ_cache_state cl_ಕಾರ್ಯಕ್ರಮ_cache_init(cl_device_id device) {
    cl_ಕಾರ್ಯಕ್ರಮ_cache_state st;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    ಆದರೆ (env && (!std::strcmp(env, "0")    || !std::strcmp(env, "off")  ||
                !std::strcmp(env, "none") || !std::strcmp(env, "disable") ||
                !std::strcmp(env, "disabled"))) {
        ಆದರೆ (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        ಹಿಂತಿರುಗಿ st;
    }

    std::string dir;
    ಆದರೆ (!env || !*env || !std::strcmp(env, "1") || !std::strcmp(env, "default")) {
        dir = default_cache_dir();
        ಆದರೆ (dir.ಖಾಲಿಗಳು()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            ಹಿಂತಿರುಗಿ st;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        dir = env;
    }

    ಆದರೆ (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        ಹಿಂತಿರುಗಿ st;
    }

    st.dir        = dir;
    st.ಕೀಲಿ_suffix = compute_ಕೀಲಿ_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    ಆದರೆ (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    ಹಿಂತಿರುಗಿ st;
}

cl_ಕಾರ್ಯಕ್ರಮ cl_ಕಾರ್ಯಕ್ರಮ_cache_try_ತುಂಬು(
    ಸ್ಥಿರಾಂಕ cl_ಕಾರ್ಯಕ್ರಮ_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                   ಮೂಲ,
    ಸ್ಥಿರಾಂಕ std::string &            compile_opts) {

    ಆದರೆ (state.dir.ಖಾಲಿಗಳು() || !ಮೂಲ) { ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ; }

    ಸ್ಥಿರಾಂಕ std::string ಕೀಲಿ  = compute_ಕೀಲಿ(state.ಕೀಲಿ_suffix, ಮೂಲ, compile_opts);
    ಸ್ಥಿರಾಂಕ std::string path = state.dir + "/" + ಕೀಲಿ + ".clbin";

    std::vector<uint8_t> file;
    ಆದರೆ (!read_all(path, file)) {
        ++g_cache_ತಪ್ಪಿದವು;
        cache_debug_ಸಾಲು("MISS", ಕೀಲಿ, ಮೂಲ, compile_opts);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (file.size() < 16 || std::memcmp(file.ದತ್ತಾಂಶ(), MAGIC, 8) != 0) { ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    ಆದರೆ (fmt != CL_ಕಾರ್ಯಕ್ರಮ_CACHE_FORMAT_ಆವೃತ್ತಿ) { ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ; }

    ಸ್ಥಿರಾಂಕ size_t hdr_len = 16;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * bin     = file.ದತ್ತಾಂಶ() + hdr_len;
    ಸ್ಥಿರಾಂಕ size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_ಕಾರ್ಯಕ್ರಮ p = cl_Create_ಕಾರ್ಯಕ್ರಮ_With_ದ್ವಿಮಾನ(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    ಆದರೆ (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == ಶೂನ್ಯಸೂಚಕ) {
        ಆದರೆ (p) { cl_Release_ಕಾರ್ಯಕ್ರಮ(p); }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    err = cl_Build_ಕಾರ್ಯಕ್ರಮ(p, 0, ಶೂನ್ಯಸೂಚಕ, compile_opts.c_str(), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ);
    ಆದರೆ (err != CL_SUCCESS) {
        cl_Release_ಕಾರ್ಯಕ್ರಮ(p);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ++g_cache_ದೊರಕಿದವು;
    cache_debug_ಸಾಲು("HIT", ಕೀಲಿ, ಮೂಲ, compile_opts);
    ಹಿಂತಿರುಗಿ p;
}

ಶೂನ್ಯ cl_ಕಾರ್ಯಕ್ರಮ_cache_try_save(
    ಸ್ಥಿರಾಂಕ cl_ಕಾರ್ಯಕ್ರಮ_cache_state & state,
    cl_ಕಾರ್ಯಕ್ರಮ                     ಕಾರ್ಯಕ್ರಮ,
    cl_device_id                   /*device*/,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                   ಮೂಲ,
    ಸ್ಥಿರಾಂಕ std::string &            compile_opts) {

    ಆದರೆ (state.dir.ಖಾಲಿಗಳು() || !ಕಾರ್ಯಕ್ರಮ || !ಮೂಲ) {
        ಹಿಂತಿರುಗಿ;
    }

    cl_uint n_dev = 0;
    ಆದರೆ (cl_Get_ಕಾರ್ಯಕ್ರಮ_Info(ಕಾರ್ಯಕ್ರಮ, CL_ಕಾರ್ಯಕ್ರಮ_ಅಂಕೆ_DEVICES, ಗಾತ್ರ(n_dev), &n_dev, ಶೂನ್ಯಸೂಚಕ) != CL_SUCCESS || n_dev == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    std::vector<size_t> sizes(n_dev);
    ಆದರೆ (cl_Get_ಕಾರ್ಯಕ್ರಮ_Info(ಕಾರ್ಯಕ್ರಮ, CL_ಕಾರ್ಯಕ್ರಮ_ದ್ವಿಮಾನ_SIZES, ಗಾತ್ರ(size_t) * n_dev, sizes.ದತ್ತಾಂಶ(), ಶೂನ್ಯಸೂಚಕ) != CL_SUCCESS) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (sizes.ಖಾಲಿಗಳು() || sizes[0] == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *>      bin_ptrs(n_dev);
    ಗಾಗಿ (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].ದತ್ತಾಂಶ();
    }
    ಆದರೆ (cl_Get_ಕಾರ್ಯಕ್ರಮ_Info(ಕಾರ್ಯಕ್ರಮ, CL_ಕಾರ್ಯಕ್ರಮ_BINARIES, ಗಾತ್ರ(ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *) * n_dev, bin_ptrs.ದತ್ತಾಂಶ(), ಶೂನ್ಯಸೂಚಕ) != CL_SUCCESS) {
        ಹಿಂತಿರುಗಿ;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & bin = binaries[0];

    std::vector<uint8_t> file;
    file.reserve(16 + bin.size());
    file.insert(file.end(), MAGIC, MAGIC + 8);
    uint32_t fmt = CL_ಕಾರ್ಯಕ್ರಮ_CACHE_FORMAT_ಆವೃತ್ತಿ;
    file.push_back((uint8_t)  (fmt        & 0xff));
    file.push_back((uint8_t) ((fmt >>  8) & 0xff));
    file.push_back((uint8_t) ((fmt >> 16) & 0xff));
    file.push_back((uint8_t) ((fmt >> 24) & 0xff));
    file.push_back(0); file.push_back(0); file.push_back(0); file.push_back(0); // reserved
    file.insert(file.end(), bin.begin(), bin.end());

    ಸ್ಥಿರಾಂಕ std::string ಕೀಲಿ  = compute_ಕೀಲಿ(state.ಕೀಲಿ_suffix, ಮೂಲ, compile_opts);
    ಸ್ಥಿರಾಂಕ std::string path = state.dir + "/" + ಕೀಲಿ + ".clbin";
    ಆದರೆ (!write_atomic(path, file.ದತ್ತಾಂಶ(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } ಇಲ್ಲದಿದ್ದರೆ {
        ++g_cache_saves;
        cache_debug_ಸಾಲು("SAVE", ಕೀಲಿ, ಮೂಲ, compile_opts);
    }
}
