// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CL_ਨਿਸ਼ਾਨਾ_OPENCL_ਸੰਸਕਰਣ GGML_OPENCL_ਨਿਸ਼ਾਨਾ_ਸੰਸਕਰਣ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ਅਨ੍ਤਰ੍ਭਾਵ "cl-program-cache.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"  // GGML_LOG_INFO / WARN

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <filesystem>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <system_error>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਜੇ defined(_WIN32)
#  ifndef WIN32_LEAN_AND_ਔਸਤ
#    define WIN32_LEAN_AND_ਔਸਤ
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((ਪੂਰਨ) GetCurrentProcessId())
#ਨਹੀਂਤਾਂ
#  include <unistd.h>
#  define ggml_getpid()    ((ਪੂਰਨ) getpid())
#ਜੇ_ਅੰਤ

ਨਾਮਾਕਾਸ਼ fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

ਨਾਮਾਕਾਸ਼ {

ਬਣਤਰ sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buf_len;
};

ਅਚਲ uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

ਸਤਰਅੰਦਰ uint32_t rotr32(uint32_t x, ਬਿਨਾਨਿਸ਼ਾਨੀ n) { ਵਾਪਸ (x >> n) | (x << (32 - n)); }

ਖਾਲੀ sha256_compress(uint32_t state[8], ਅਚਲ uint8_t ਖੰਡ[64]) {
    uint32_t w[64];
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)ਖੰਡ[i*4    ] << 24) |
               ((uint32_t)ਖੰਡ[i*4 + 1] << 16) |
               ((uint32_t)ਖੰਡ[i*4 + 2] <<  8) |
               ((uint32_t)ਖੰਡ[i*4 + 3]      );
    }
    ਲਈ (ਪੂਰਨ i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    ਲਈ (ਪੂਰਨ i = 0; i < 64; ++i) {
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

ਖਾਲੀ sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

ਖਾਲੀ sha256_update(sha256_ctx & c, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t len) {
    ਅਚਲ uint8_t * p = (ਅਚਲ uint8_t *) ਅੰਕੜੇ;
    c.bitlen += (uint64_t) len * 8;
    ਜੇ (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        ਜੇ (n > len) { n = len; }
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        ਜੇ (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    ਜਦੋਂ (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    ਜੇ (len > 0) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(c.buf, p, len);
        c.buf_len = len;
    }
}

ਖਾਲੀ sha256_final(sha256_ctx & c, uint8_t ਬਾਹਰਲਾ[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    ਜੇ (c.buf_len > 56) {
        ਜਦੋਂ (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    ਜਦੋਂ (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    ਲਈ (ਪੂਰਨ i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        ਬਾਹਰਲਾ[i*4    ] = (uint8_t) (c.state[i] >> 24);
        ਬਾਹਰਲਾ[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        ਬਾਹਰਲਾ[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        ਬਾਹਰਲਾ[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(ਅਚਲ uint8_t digest[32]) {
    ਸਥਿਰ ਅਚਲ ਅੱਖਰ hex[] = "0123456789abcdef";
    std::string s(64, '0');
    ਲਈ (ਪੂਰਨ i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    ਵਾਪਸ s;
}

std::string compute_ਕੁੰਜੀ(ਅਚਲ std::string & ਕੁੰਜੀ_suffix,
                        ਅਚਲ ਅੱਖਰ *        ਸਰੋਤ,
                        ਅਚਲ std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    ਸਥਿਰ ਅਚਲ uint8_t sep = 0;
    sha256_update(c, ਸਰੋਤ,             ਲੜੀ_ਲੰਬਾਈ(ਸਰੋਤ));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.ਅੰਕੜੇ(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, ਕੁੰਜੀ_suffix.ਅੰਕੜੇ(),  ਕੁੰਜੀ_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    ਵਾਪਸ sha256_hex(digest);
}

ਤਾਰਕਿਕ make_dir_recursive(ਅਚਲ std::string & path) {
    ਜੇ (path.ਖਾਲੀ_ਥਾਵਾਂ()) { ਵਾਪਸ ਝੂਠ; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    ਅਚਲ fs::path p = fs::u8path(path);
    std::error_code ec;
    ਜੇ (fs::create_directories(p, ec)) { ਵਾਪਸ ਸੱਚ; }
    std::error_code ec_stat;
    ਵਾਪਸ fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#ਜੇ defined(_WIN32)
    ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = std::getenv("LOCALAPPDATA");
    ਜੇ (!ਅਧਾਰ || !*ਅਧਾਰ) { ਅਧਾਰ = std::getenv("APPDATA"); }
    ਜੇ (!ਅਧਾਰ || !*ਅਧਾਰ) { ਅਧਾਰ = std::getenv("TEMP"); }
    ਜੇ (!ਅਧਾਰ || !*ਅਧਾਰ) { ਅਧਾਰ = "."; }
    ਵਾਪਸ std::string(ਅਧਾਰ) + "\\llama.cpp\\cl-cache";
#ਜਾਂਜੇ defined(__APPLE__)
    ਅਚਲ ਅੱਖਰ * home = std::getenv("HOME");
    ਜੇ (!home || !*home) { home = "."; }
    ਵਾਪਸ std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#ਨਹੀਂਤਾਂ
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    ਅਚਲ fs::path tmp_ਰਸਤਾ = fs::temp_directory_path(ec);
    ਜੇ (ec || tmp_ਰਸਤਾ.ਖਾਲੀ_ਥਾਵਾਂ()) { ਵਾਪਸ {}; }
    ਵਾਪਸ tmp_ਰਸਤਾ.string() + "/llama.cpp/cl-cache";
#ਜੇ_ਅੰਤ
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ GetInfoFn, ਕਿਸਮ_ਨਾਂ Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    ਜੇ (fn(obj, name, 0, ਖਾਲੀ_ਸੂਚਕ, &sz) != CL_SUCCESS || sz == 0) {
        ਵਾਪਸ {};
    }
    std::string s(sz, '\0');
    ਜੇ (fn(obj, name, sz, &s[0], ਖਾਲੀ_ਸੂਚਕ) != CL_SUCCESS) {
        ਵਾਪਸ {};
    }
    ਜੇ (!s.ਖਾਲੀ_ਥਾਵਾਂ() && s.back() == '\0') {
        s.pop_back();
    }
    ਵਾਪਸ s;
}

std::string compute_ਕੁੰਜੀ_suffix(cl_device_id device) {
    cl_platform_id platform = ਖਾਲੀ_ਸੂਚਕ;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, ਆਕਾਰ(platform), &platform, ਖਾਲੀ_ਸੂਚਕ);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_ਸੰਸਕਰਣ);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_ਸੰਸਕਰਣ);    s.push_back('\0');
    ਜੇ (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_ਸੰਸਕਰਣ); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_ਪ੍ਰੋਗਰਾਮ_CACHE_FORMAT_ਸੰਸਕਰਣ);
    ਵਾਪਸ s;
}

ਅਚਲ uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

ਤਾਰਕਿਕ read_all(ਅਚਲ std::string & path, std::vector<uint8_t> & ਬਾਹਰਲਾ) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    ਜੇ (!f) { ਵਾਪਸ ਝੂਠ; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    ਜੇ (sz < 0) { ਵਾਪਸ ਝੂਠ; }
    f.seekg(0, std::ios::beg);
    ਬਾਹਰਲਾ.resize((size_t) sz);
    ਜੇ (sz > 0) { f.ਪਠ((ਅੱਖਰ *) ਬਾਹਰਲਾ.ਅੰਕੜੇ(), sz); }
    ਵਾਪਸ f.good() || f.eof();
}

ਤਾਰਕਿਕ write_atomic(ਅਚਲ std::string & path, ਅਚਲ uint8_t * ਅੰਕੜੇ, size_t len) {
    ਅਚਲ fs::path dst = fs::u8path(path);
    ਅਚਲ fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        ਜੇ (!f) { ਵਾਪਸ ਝੂਠ; }
        f.ਲਿਖ((ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ, (std::streamsize) len);
        ਜੇ (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    ਜੇ (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

}  // namespace

ਸਥਿਰ ਤਾਰਕਿਕ cache_debug_enabled() {
    ਸਥਿਰ ਪੂਰਨ cached = -1;
    ਜੇ (cached < 0) {
        ਅਚਲ ਅੱਖਰ * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    ਵਾਪਸ cached != 0;
}

ਸਥਿਰ std::string opts_preview(ਅਚਲ std::string & opts, size_t n = 120) {
    ਜੇ (opts.size() <= n) { ਵਾਪਸ opts; }
    ਵਾਪਸ opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
ਸਥਿਰ ਪੂਰਨ g_cache_ਪ੍ਰਾਪਤੀਆਂ = 0, g_cache_ਖੁੰਝਾਂ = 0, g_cache_saves = 0;

// Debug trace directly to stderr
ਸਥਿਰ ਖਾਲੀ cache_debug_ਸਤਰ(ਅਚਲ ਅੱਖਰ * ਕਿਸਮ_2, ਅਚਲ std::string & ਕੁੰਜੀ,
                             ਅਚਲ ਅੱਖਰ * ਸਰੋਤ, ਅਚਲ std::string & opts) {
    ਜੇ (!cache_debug_enabled()) { ਵਾਪਸ; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            ਕਿਸਮ_2, g_cache_ਪ੍ਰਾਪਤੀਆਂ, g_cache_ਖੁੰਝਾਂ, g_cache_saves,
            ਕੁੰਜੀ.substr(0, 16).c_str(), ਲੜੀ_ਲੰਬਾਈ(ਸਰੋਤ), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_ਪ੍ਰੋਗਰਾਮ_cache_state cl_ਪ੍ਰੋਗਰਾਮ_cache_init(cl_device_id device) {
    cl_ਪ੍ਰੋਗਰਾਮ_cache_state st;

    ਅਚਲ ਅੱਖਰ * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    ਜੇ (env && (!std::strcmp(env, "0")    || !std::strcmp(env, "off")  ||
                !std::strcmp(env, "none") || !std::strcmp(env, "disable") ||
                !std::strcmp(env, "disabled"))) {
        ਜੇ (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        ਵਾਪਸ st;
    }

    std::string dir;
    ਜੇ (!env || !*env || !std::strcmp(env, "1") || !std::strcmp(env, "default")) {
        dir = default_cache_dir();
        ਜੇ (dir.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            ਵਾਪਸ st;
        }
    } ਨਹੀਂਤਾਂ {
        dir = env;
    }

    ਜੇ (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        ਵਾਪਸ st;
    }

    st.dir        = dir;
    st.ਕੁੰਜੀ_suffix = compute_ਕੁੰਜੀ_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    ਜੇ (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    ਵਾਪਸ st;
}

cl_ਪ੍ਰੋਗਰਾਮ cl_ਪ੍ਰੋਗਰਾਮ_cache_try_ਲੱਦੋ(
    ਅਚਲ cl_ਪ੍ਰੋਗਰਾਮ_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    ਅਚਲ ਅੱਖਰ *                   ਸਰੋਤ,
    ਅਚਲ std::string &            compile_opts) {

    ਜੇ (state.dir.ਖਾਲੀ_ਥਾਵਾਂ() || !ਸਰੋਤ) { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }

    ਅਚਲ std::string ਕੁੰਜੀ  = compute_ਕੁੰਜੀ(state.ਕੁੰਜੀ_suffix, ਸਰੋਤ, compile_opts);
    ਅਚਲ std::string path = state.dir + "/" + ਕੁੰਜੀ + ".clbin";

    std::vector<uint8_t> file;
    ਜੇ (!read_all(path, file)) {
        ++g_cache_ਖੁੰਝਾਂ;
        cache_debug_ਸਤਰ("MISS", ਕੁੰਜੀ, ਸਰੋਤ, compile_opts);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (file.size() < 16 || std::memcmp(file.ਅੰਕੜੇ(), MAGIC, 8) != 0) { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    ਜੇ (fmt != CL_ਪ੍ਰੋਗਰਾਮ_CACHE_FORMAT_ਸੰਸਕਰਣ) { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }

    ਅਚਲ size_t hdr_len = 16;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * bin     = file.ਅੰਕੜੇ() + hdr_len;
    ਅਚਲ size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_ਪ੍ਰੋਗਰਾਮ p = cl_Create_ਪ੍ਰੋਗਰਾਮ_With_ਦੁਆਧਾਰੀ(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    ਜੇ (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == ਖਾਲੀ_ਸੂਚਕ) {
        ਜੇ (p) { cl_Release_ਪ੍ਰੋਗਰਾਮ(p); }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    err = cl_Build_ਪ੍ਰੋਗਰਾਮ(p, 0, ਖਾਲੀ_ਸੂਚਕ, compile_opts.c_str(), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ);
    ਜੇ (err != CL_SUCCESS) {
        cl_Release_ਪ੍ਰੋਗਰਾਮ(p);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ++g_cache_ਪ੍ਰਾਪਤੀਆਂ;
    cache_debug_ਸਤਰ("HIT", ਕੁੰਜੀ, ਸਰੋਤ, compile_opts);
    ਵਾਪਸ p;
}

ਖਾਲੀ cl_ਪ੍ਰੋਗਰਾਮ_cache_try_save(
    ਅਚਲ cl_ਪ੍ਰੋਗਰਾਮ_cache_state & state,
    cl_ਪ੍ਰੋਗਰਾਮ                     ਪ੍ਰੋਗਰਾਮ,
    cl_device_id                   /*device*/,
    ਅਚਲ ਅੱਖਰ *                   ਸਰੋਤ,
    ਅਚਲ std::string &            compile_opts) {

    ਜੇ (state.dir.ਖਾਲੀ_ਥਾਵਾਂ() || !ਪ੍ਰੋਗਰਾਮ || !ਸਰੋਤ) {
        ਵਾਪਸ;
    }

    cl_uint n_dev = 0;
    ਜੇ (cl_Get_ਪ੍ਰੋਗਰਾਮ_Info(ਪ੍ਰੋਗਰਾਮ, CL_ਪ੍ਰੋਗਰਾਮ_ਅੰਕ_DEVICES, ਆਕਾਰ(n_dev), &n_dev, ਖਾਲੀ_ਸੂਚਕ) != CL_SUCCESS || n_dev == 0) {
        ਵਾਪਸ;
    }

    std::vector<size_t> sizes(n_dev);
    ਜੇ (cl_Get_ਪ੍ਰੋਗਰਾਮ_Info(ਪ੍ਰੋਗਰਾਮ, CL_ਪ੍ਰੋਗਰਾਮ_ਦੁਆਧਾਰੀ_SIZES, ਆਕਾਰ(size_t) * n_dev, sizes.ਅੰਕੜੇ(), ਖਾਲੀ_ਸੂਚਕ) != CL_SUCCESS) {
        ਵਾਪਸ;
    }
    ਜੇ (sizes.ਖਾਲੀ_ਥਾਵਾਂ() || sizes[0] == 0) {
        ਵਾਪਸ;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *>      bin_ptrs(n_dev);
    ਲਈ (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].ਅੰਕੜੇ();
    }
    ਜੇ (cl_Get_ਪ੍ਰੋਗਰਾਮ_Info(ਪ੍ਰੋਗਰਾਮ, CL_ਪ੍ਰੋਗਰਾਮ_BINARIES, ਆਕਾਰ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *) * n_dev, bin_ptrs.ਅੰਕੜੇ(), ਖਾਲੀ_ਸੂਚਕ) != CL_SUCCESS) {
        ਵਾਪਸ;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    ਅਚਲ std::vector<uint8_t> & bin = binaries[0];

    std::vector<uint8_t> file;
    file.reserve(16 + bin.size());
    file.insert(file.end(), MAGIC, MAGIC + 8);
    uint32_t fmt = CL_ਪ੍ਰੋਗਰਾਮ_CACHE_FORMAT_ਸੰਸਕਰਣ;
    file.push_back((uint8_t)  (fmt        & 0xff));
    file.push_back((uint8_t) ((fmt >>  8) & 0xff));
    file.push_back((uint8_t) ((fmt >> 16) & 0xff));
    file.push_back((uint8_t) ((fmt >> 24) & 0xff));
    file.push_back(0); file.push_back(0); file.push_back(0); file.push_back(0); // reserved
    file.insert(file.end(), bin.begin(), bin.end());

    ਅਚਲ std::string ਕੁੰਜੀ  = compute_ਕੁੰਜੀ(state.ਕੁੰਜੀ_suffix, ਸਰੋਤ, compile_opts);
    ਅਚਲ std::string path = state.dir + "/" + ਕੁੰਜੀ + ".clbin";
    ਜੇ (!write_atomic(path, file.ਅੰਕੜੇ(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } ਨਹੀਂਤਾਂ {
        ++g_cache_saves;
        cache_debug_ਸਤਰ("SAVE", ਕੁੰਜੀ, ਸਰੋਤ, compile_opts);
    }
}
