// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#அந்தர்பாவ "cl-program-cache.h"

#அந்தர்பாவ "ggml-impl.h"  // GGML_LOG_INFO / WARN

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <filesystem>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <system_error>
#அந்தர்பாவ <vector>

#எனில் defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((முழு) GetCurrentProcessId())
#இல்லையெனில்
#  include <unistd.h>
#  define ggml_getpid()    ((முழு) getpid())
#endif

நாமாகாஶ fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

நாமாகாஶ {

கட்டமைப்பு sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buf_len;
};

மாறிலி uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

உள்வரி uint32_t rotr32(uint32_t x, unsigned n) { திரும்பு (x >> n) | (x << (32 - n)); }

வெறுமை sha256_compress(uint32_t state[8], மாறிலி uint8_t block[64]) {
    uint32_t w[64];
    க்கு (முழு i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4    ] << 24) |
               ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] <<  8) |
               ((uint32_t)block[i*4 + 3]      );
    }
    க்கு (முழு i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    க்கு (முழு i = 0; i < 64; ++i) {
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

வெறுமை sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

வெறுமை sha256_update(sha256_ctx & c, மாறிலி வெறுமை * data, size_t len) {
    மாறிலி uint8_t * p = (மாறிலி uint8_t *) data;
    c.bitlen += (uint64_t) len * 8;
    எனில் (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        எனில் (n > len) { n = len; }
        memcpy(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        எனில் (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    போது (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    எனில் (len > 0) {
        memcpy(c.buf, p, len);
        c.buf_len = len;
    }
}

வெறுமை sha256_final(sha256_ctx & c, uint8_t out[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    எனில் (c.buf_len > 56) {
        போது (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    போது (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    க்கு (முழு i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    க்கு (முழு i = 0; i < 8; ++i) {
        out[i*4    ] = (uint8_t) (c.state[i] >> 24);
        out[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        out[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        out[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(மாறிலி uint8_t digest[32]) {
    நிலையான மாறிலி எழுத்து hex[] = "0123456789abcdef";
    std::string s(64, '0');
    க்கு (முழு i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    திரும்பு s;
}

std::string compute_சாவி(மாறிலி std::string & சாவி_suffix,
                        மாறிலி எழுத்து *        source,
                        மாறிலி std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    நிலையான மாறிலி uint8_t sep = 0;
    sha256_update(c, source,             strlen(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.data(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, சாவி_suffix.data(),  சாவி_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    திரும்பு sha256_hex(digest);
}

தருக்க make_dir_recursive(மாறிலி std::string & path) {
    எனில் (path.empty()) { திரும்பு பொய்; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    மாறிலி fs::path p = fs::u8path(path);
    std::error_code ec;
    எனில் (fs::create_directories(p, ec)) { திரும்பு உண்மை; }
    std::error_code ec_stat;
    திரும்பு fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#எனில் defined(_WIN32)
    மாறிலி எழுத்து * base = std::getenv("LOCALAPPDATA");
    எனில் (!base || !*base) { base = std::getenv("APPDATA"); }
    எனில் (!base || !*base) { base = std::getenv("TEMP"); }
    எனில் (!base || !*base) { base = "."; }
    திரும்பு std::string(base) + "\\llama.cpp\\cl-cache";
#அல்லதெனில் defined(__APPLE__)
    மாறிலி எழுத்து * home = std::getenv("HOME");
    எனில் (!home || !*home) { home = "."; }
    திரும்பு std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#இல்லையெனில்
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    மாறிலி fs::path tmp_path = fs::temp_directory_path(ec);
    எனில் (ec || tmp_path.empty()) { திரும்பு {}; }
    திரும்பு tmp_path.string() + "/llama.cpp/cl-cache";
#endif
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
template <typename GetInfoFn, typename Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    எனில் (fn(obj, name, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) {
        திரும்பு {};
    }
    std::string s(sz, '\0');
    எனில் (fn(obj, name, sz, &s[0], nullptr) != CL_SUCCESS) {
        திரும்பு {};
    }
    எனில் (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    திரும்பு s;
}

std::string compute_சாவி_suffix(cl_device_id device) {
    cl_platform_id platform = nullptr;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, அளவு(platform), &platform, nullptr);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_VERSION);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_VERSION);    s.push_back('\0');
    எனில் (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_VERSION); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_PROGRAM_CACHE_FORMAT_VERSION);
    திரும்பு s;
}

மாறிலி uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

தருக்க read_all(மாறிலி std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    எனில் (!f) { திரும்பு பொய்; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    எனில் (sz < 0) { திரும்பு பொய்; }
    f.seekg(0, std::ios::beg);
    out.resize((size_t) sz);
    எனில் (sz > 0) { f.பட((எழுத்து *) out.data(), sz); }
    திரும்பு f.good() || f.eof();
}

தருக்க write_atomic(மாறிலி std::string & path, மாறிலி uint8_t * data, size_t len) {
    மாறிலி fs::path dst = fs::u8path(path);
    மாறிலி fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        எனில் (!f) { திரும்பு பொய்; }
        f.லிக((மாறிலி எழுத்து *) data, (std::streamsize) len);
        எனில் (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            திரும்பு பொய்;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    எனில் (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

}  // namespace

நிலையான தருக்க cache_debug_enabled() {
    நிலையான முழு cached = -1;
    எனில் (cached < 0) {
        மாறிலி எழுத்து * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    திரும்பு cached != 0;
}

நிலையான std::string opts_preview(மாறிலி std::string & opts, size_t n = 120) {
    எனில் (opts.size() <= n) { திரும்பு opts; }
    திரும்பு opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
நிலையான முழு g_cache_hits = 0, g_cache_misses = 0, g_cache_saves = 0;

// Debug trace directly to stderr
நிலையான வெறுமை cache_debug_வரி(மாறிலி எழுத்து * kind, மாறிலி std::string & சாவி,
                             மாறிலி எழுத்து * source, மாறிலி std::string & opts) {
    எனில் (!cache_debug_enabled()) { திரும்பு; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            kind, g_cache_hits, g_cache_misses, g_cache_saves,
            சாவி.substr(0, 16).c_str(), strlen(source), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_program_cache_state cl_program_cache_init(cl_device_id device) {
    cl_program_cache_state st;

    மாறிலி எழுத்து * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    எனில் (env && (!std::strcmp(env, "0")    || !std::strcmp(env, "off")  ||
                !std::strcmp(env, "none") || !std::strcmp(env, "disable") ||
                !std::strcmp(env, "disabled"))) {
        எனில் (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        திரும்பு st;
    }

    std::string dir;
    எனில் (!env || !*env || !std::strcmp(env, "1") || !std::strcmp(env, "default")) {
        dir = default_cache_dir();
        எனில் (dir.empty()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            திரும்பு st;
        }
    } இல்லையெனில் {
        dir = env;
    }

    எனில் (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        திரும்பு st;
    }

    st.dir        = dir;
    st.சாவி_suffix = compute_சாவி_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    எனில் (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    திரும்பு st;
}

cl_program cl_program_cache_try_load(
    மாறிலி cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    மாறிலி எழுத்து *                   source,
    மாறிலி std::string &            compile_opts) {

    எனில் (state.dir.empty() || !source) { திரும்பு nullptr; }

    மாறிலி std::string சாவி  = compute_சாவி(state.சாவி_suffix, source, compile_opts);
    மாறிலி std::string path = state.dir + "/" + சாவி + ".clbin";

    std::vector<uint8_t> file;
    எனில் (!read_all(path, file)) {
        ++g_cache_misses;
        cache_debug_வரி("MISS", சாவி, source, compile_opts);
        திரும்பு nullptr;
    }
    எனில் (file.size() < 16 || std::memcmp(file.data(), MAGIC, 8) != 0) { திரும்பு nullptr; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    எனில் (fmt != CL_PROGRAM_CACHE_FORMAT_VERSION) { திரும்பு nullptr; }

    மாறிலி size_t hdr_len = 16;
    மாறிலி unsigned எழுத்து * bin     = file.data() + hdr_len;
    மாறிலி size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_program p = cl_Create_Program_With_இருமம்(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    எனில் (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == nullptr) {
        எனில் (p) { clReleaseProgram(p); }
        திரும்பு nullptr;
    }

    err = clBuildProgram(p, 0, nullptr, compile_opts.c_str(), nullptr, nullptr);
    எனில் (err != CL_SUCCESS) {
        clReleaseProgram(p);
        திரும்பு nullptr;
    }
    ++g_cache_hits;
    cache_debug_வரி("HIT", சாவி, source, compile_opts);
    திரும்பு p;
}

வெறுமை cl_program_cache_try_save(
    மாறிலி cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   /*device*/,
    மாறிலி எழுத்து *                   source,
    மாறிலி std::string &            compile_opts) {

    எனில் (state.dir.empty() || !program || !source) {
        திரும்பு;
    }

    cl_uint n_dev = 0;
    எனில் (clGetProgramInfo(program, CL_PROGRAM_இலக்கம்_DEVICES, அளவு(n_dev), &n_dev, nullptr) != CL_SUCCESS || n_dev == 0) {
        திரும்பு;
    }

    std::vector<size_t> sizes(n_dev);
    எனில் (clGetProgramInfo(program, CL_PROGRAM_இருமம்_SIZES, அளவு(size_t) * n_dev, sizes.data(), nullptr) != CL_SUCCESS) {
        திரும்பு;
    }
    எனில் (sizes.empty() || sizes[0] == 0) {
        திரும்பு;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<unsigned எழுத்து *>      bin_ptrs(n_dev);
    க்கு (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].data();
    }
    எனில் (clGetProgramInfo(program, CL_PROGRAM_BINARIES, அளவு(unsigned எழுத்து *) * n_dev, bin_ptrs.data(), nullptr) != CL_SUCCESS) {
        திரும்பு;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    மாறிலி std::vector<uint8_t> & bin = binaries[0];

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

    மாறிலி std::string சாவி  = compute_சாவி(state.சாவி_suffix, source, compile_opts);
    மாறிலி std::string path = state.dir + "/" + சாவி + ".clbin";
    எனில் (!write_atomic(path, file.data(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } இல்லையெனில் {
        ++g_cache_saves;
        cache_debug_வரி("SAVE", சாவி, source, compile_opts);
    }
}
