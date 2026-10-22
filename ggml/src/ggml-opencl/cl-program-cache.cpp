// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ضمن "cl-program-cache.h"

#ضمن "ggml-impl.h"  // GGML_LOG_INFO / WARN

#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <filesystem>
#ضمن <fstream>
#ضمن <system_error>
#ضمن <vector>

#إذا defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((صحيح) GetCurrentProcessId())
#وإلا
#  include <unistd.h>
#  define ggml_getpid()    ((صحيح) getpid())
#endif

نطاق fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

نطاق {

بنية sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buf_len;
};

ثابت uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

مضمن uint32_t rotr32(uint32_t x, unsigned n) { أرجع (x >> n) | (x << (32 - n)); }

فراغ sha256_compress(uint32_t state[8], ثابت uint8_t block[64]) {
    uint32_t w[64];
    لكل (صحيح i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4    ] << 24) |
               ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] <<  8) |
               ((uint32_t)block[i*4 + 3]      );
    }
    لكل (صحيح i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    لكل (صحيح i = 0; i < 64; ++i) {
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

فراغ sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

فراغ sha256_update(sha256_ctx & c, ثابت فراغ * data, size_t len) {
    ثابت uint8_t * p = (ثابت uint8_t *) data;
    c.bitlen += (uint64_t) len * 8;
    إذا (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        إذا (n > len) { n = len; }
        memcpy(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        إذا (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    طالما (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    إذا (len > 0) {
        memcpy(c.buf, p, len);
        c.buf_len = len;
    }
}

فراغ sha256_final(sha256_ctx & c, uint8_t out[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    إذا (c.buf_len > 56) {
        طالما (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    طالما (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    لكل (صحيح i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    لكل (صحيح i = 0; i < 8; ++i) {
        out[i*4    ] = (uint8_t) (c.state[i] >> 24);
        out[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        out[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        out[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(ثابت uint8_t digest[32]) {
    ساكن ثابت حرف hex[] = "0123456789abcdef";
    std::string s(64, '0');
    لكل (صحيح i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    أرجع s;
}

std::string compute_کونچي(ثابت std::string & کونچي_suffix,
                        ثابت حرف *        source,
                        ثابت std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    ساكن ثابت uint8_t sep = 0;
    sha256_update(c, source,             strlen(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.data(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, کونچي_suffix.data(),  کونچي_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    أرجع sha256_hex(digest);
}

منطقي make_dir_recursive(ثابت std::string & path) {
    إذا (path.empty()) { أرجع خطأ; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    ثابت fs::path p = fs::u8path(path);
    std::error_code ec;
    إذا (fs::create_directories(p, ec)) { أرجع true; }
    std::error_code ec_stat;
    أرجع fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#إذا defined(_WIN32)
    ثابت حرف * base = std::getenv("LOCALAPPDATA");
    إذا (!base || !*base) { base = std::getenv("APPDATA"); }
    إذا (!base || !*base) { base = std::getenv("TEMP"); }
    إذا (!base || !*base) { base = "."; }
    أرجع std::string(base) + "\\llama.cpp\\cl-cache";
#وإلاإذا defined(__APPLE__)
    ثابت حرف * home = std::getenv("HOME");
    إذا (!home || !*home) { home = "."; }
    أرجع std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#وإلا
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    ثابت fs::path tmp_path = fs::temp_directory_path(ec);
    إذا (ec || tmp_path.empty()) { أرجع {}; }
    أرجع tmp_path.string() + "/llama.cpp/cl-cache";
#endif
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
template <typename GetInfoFn, typename Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    إذا (fn(obj, name, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) {
        أرجع {};
    }
    std::string s(sz, '\0');
    إذا (fn(obj, name, sz, &s[0], nullptr) != CL_SUCCESS) {
        أرجع {};
    }
    إذا (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    أرجع s;
}

std::string compute_کونچي_suffix(cl_device_id device) {
    cl_platform_id platform = nullptr;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, حجم(platform), &platform, nullptr);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_VERSION);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_VERSION);    s.push_back('\0');
    إذا (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_VERSION); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_PROGRAM_CACHE_FORMAT_VERSION);
    أرجع s;
}

ثابت uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

منطقي read_all(ثابت std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    إذا (!f) { أرجع خطأ; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    إذا (sz < 0) { أرجع خطأ; }
    f.seekg(0, std::ios::beg);
    out.resize((size_t) sz);
    إذا (sz > 0) { f.اقرأ((حرف *) out.data(), sz); }
    أرجع f.good() || f.eof();
}

منطقي write_atomic(ثابت std::string & path, ثابت uint8_t * data, size_t len) {
    ثابت fs::path dst = fs::u8path(path);
    ثابت fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        إذا (!f) { أرجع خطأ; }
        f.اكتب((ثابت حرف *) data, (std::streamsize) len);
        إذا (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            أرجع خطأ;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    إذا (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        أرجع خطأ;
    }
    أرجع true;
}

}  // namespace

ساكن منطقي cache_debug_enabled() {
    ساكن صحيح cached = -1;
    إذا (cached < 0) {
        ثابت حرف * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    أرجع cached != 0;
}

ساكن std::string opts_preview(ثابت std::string & opts, size_t n = 120) {
    إذا (opts.size() <= n) { أرجع opts; }
    أرجع opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
ساكن صحيح g_cache_hits = 0, g_cache_misses = 0, g_cache_saves = 0;

// Debug trace directly to stderr
ساكن فراغ cache_debug_باريس(ثابت حرف * kind, ثابت std::string & کونچي,
                             ثابت حرف * source, ثابت std::string & opts) {
    إذا (!cache_debug_enabled()) { أرجع; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            kind, g_cache_hits, g_cache_misses, g_cache_saves,
            کونچي.substr(0, 16).c_str(), strlen(source), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_program_cache_state cl_program_cache_init(cl_device_id device) {
    cl_program_cache_state st;

    ثابت حرف * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    إذا (env && (!std::strcmp(env, "0")    || !std::strcmp(env, "off")  ||
                !std::strcmp(env, "none") || !std::strcmp(env, "disable") ||
                !std::strcmp(env, "disabled"))) {
        إذا (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        أرجع st;
    }

    std::string dir;
    إذا (!env || !*env || !std::strcmp(env, "1") || !std::strcmp(env, "default")) {
        dir = default_cache_dir();
        إذا (dir.empty()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            أرجع st;
        }
    } وإلا {
        dir = env;
    }

    إذا (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        أرجع st;
    }

    st.dir        = dir;
    st.کونچي_suffix = compute_کونچي_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    إذا (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    أرجع st;
}

cl_program cl_program_cache_try_load(
    ثابت cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    ثابت حرف *                   source,
    ثابت std::string &            compile_opts) {

    إذا (state.dir.empty() || !source) { أرجع nullptr; }

    ثابت std::string کونچي  = compute_کونچي(state.کونچي_suffix, source, compile_opts);
    ثابت std::string path = state.dir + "/" + کونچي + ".clbin";

    std::vector<uint8_t> file;
    إذا (!read_all(path, file)) {
        ++g_cache_misses;
        cache_debug_باريس("MISS", کونچي, source, compile_opts);
        أرجع nullptr;
    }
    إذا (file.size() < 16 || std::memcmp(file.data(), MAGIC, 8) != 0) { أرجع nullptr; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    إذا (fmt != CL_PROGRAM_CACHE_FORMAT_VERSION) { أرجع nullptr; }

    ثابت size_t hdr_len = 16;
    ثابت unsigned حرف * bin     = file.data() + hdr_len;
    ثابت size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_program p = cl_Create_Program_With_بيناري(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    إذا (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == nullptr) {
        إذا (p) { clReleaseProgram(p); }
        أرجع nullptr;
    }

    err = clBuildProgram(p, 0, nullptr, compile_opts.c_str(), nullptr, nullptr);
    إذا (err != CL_SUCCESS) {
        clReleaseProgram(p);
        أرجع nullptr;
    }
    ++g_cache_hits;
    cache_debug_باريس("HIT", کونچي, source, compile_opts);
    أرجع p;
}

فراغ cl_program_cache_try_save(
    ثابت cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   /*device*/,
    ثابت حرف *                   source,
    ثابت std::string &            compile_opts) {

    إذا (state.dir.empty() || !program || !source) {
        أرجع;
    }

    cl_uint n_dev = 0;
    إذا (clGetProgramInfo(program, CL_PROGRAM_اڠک_DEVICES, حجم(n_dev), &n_dev, nullptr) != CL_SUCCESS || n_dev == 0) {
        أرجع;
    }

    std::vector<size_t> sizes(n_dev);
    إذا (clGetProgramInfo(program, CL_PROGRAM_بيناري_SIZES, حجم(size_t) * n_dev, sizes.data(), nullptr) != CL_SUCCESS) {
        أرجع;
    }
    إذا (sizes.empty() || sizes[0] == 0) {
        أرجع;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<unsigned حرف *>      bin_ptrs(n_dev);
    لكل (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].data();
    }
    إذا (clGetProgramInfo(program, CL_PROGRAM_BINARIES, حجم(unsigned حرف *) * n_dev, bin_ptrs.data(), nullptr) != CL_SUCCESS) {
        أرجع;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    ثابت std::vector<uint8_t> & bin = binaries[0];

    std::vector<uint8_t> file;
    file.reserve(16 + bin.size());
    file.insert(file.end(), MAGIC, MAGIC + 8);
    uint32_t fmt = CL_PROGRAM_CACHE_FORMAT_VERSION;
    file.push_back((uint8_t)  (fmt        & 0xff));
    file.push_back((uint8_t) ((fmt >>  8) & 0xff));
    file.push_back((uint8_t) ((fmt >> 16) & 0xff));
    file.push_back((uint8_t) ((fmt >> 24) & 0xff));
    file.push_back(0); file.push_back(0); file.push_back(0); file.push_back(0); // reserved
    file.insert(file.end(), bin.begin(), bin.end());

    ثابت std::string کونچي  = compute_کونچي(state.کونچي_suffix, source, compile_opts);
    ثابت std::string path = state.dir + "/" + کونچي + ".clbin";
    إذا (!write_atomic(path, file.data(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } وإلا {
        ++g_cache_saves;
        cache_debug_باريس("SAVE", کونچي, source, compile_opts);
    }
}
