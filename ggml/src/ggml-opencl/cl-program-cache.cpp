// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#घोषणा CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#घोषणा CL_USE_DEPRECATED_OPENCL_1_2_APIS

#समावेश "cl-program-cache.h"

#समावेश "ggml-impl.h"  // GGML_LOG_INFO / WARN

#समावेश <cstdint>
#समावेश <cstdio>
#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <filesystem>
#समावेश <fstream>
#समावेश <system_error>
#समावेश <vector>

#अगर defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((पूर्णांक) GetCurrentProcessId())
#अन्यथा
#  include <unistd.h>
#  define ggml_getpid()    ((पूर्णांक) getpid())
#पूर्ण_अगर

namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

namespace {

काष्ठा sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    माप_प्रकार   buf_len;
};

स्थिर uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

अंतरभूत uint32_t rotr32(uint32_t x, अचिन्हित n) { वापस (x >> n) | (x << (32 - n)); }

व्योम sha256_compress(uint32_t state[8], स्थिर uint8_t block[64]) {
    uint32_t w[64];
    क्रम (पूर्णांक i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4    ] << 24) |
               ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] <<  8) |
               ((uint32_t)block[i*4 + 3]      );
    }
    क्रम (पूर्णांक i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    क्रम (पूर्णांक i = 0; i < 64; ++i) {
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

व्योम sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

व्योम sha256_update(sha256_ctx & c, स्थिर व्योम * तथ्य, माप_प्रकार len) {
    स्थिर uint8_t * p = (स्थिर uint8_t *) तथ्य;
    c.bitlen += (uint64_t) len * 8;
    अगर (c.buf_len > 0) {
        माप_प्रकार n = 64 - c.buf_len;
        अगर (n > len) { n = len; }
        स_नकल(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        अगर (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    जबतक (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    अगर (len > 0) {
        स_नकल(c.buf, p, len);
        c.buf_len = len;
    }
}

व्योम sha256_final(sha256_ctx & c, uint8_t out[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    अगर (c.buf_len > 56) {
        जबतक (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    जबतक (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    क्रम (पूर्णांक i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    क्रम (पूर्णांक i = 0; i < 8; ++i) {
        out[i*4    ] = (uint8_t) (c.state[i] >> 24);
        out[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        out[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        out[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(स्थिर uint8_t digest[32]) {
    अटल स्थिर अक्षर षष्ठ[] = "0123456789abcdef";
    std::string s(64, '0');
    क्रम (पूर्णांक i = 0; i < 32; ++i) {
        s[i*2    ] = षष्ठ[digest[i] >> 4];
        s[i*2 + 1] = षष्ठ[digest[i] & 0xf];
    }
    वापस s;
}

std::string compute_कुंजी(स्थिर std::string & कुंजी_suffix,
                        स्थिर अक्षर *        source,
                        स्थिर std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    अटल स्थिर uint8_t sep = 0;
    sha256_update(c, source,             म_माप(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.तथ्य(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, कुंजी_suffix.तथ्य(),  कुंजी_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    वापस sha256_hex(digest);
}

bool make_dir_recursive(स्थिर std::string & path) {
    अगर (path.empty()) { वापस false; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    स्थिर fs::path p = fs::u8path(path);
    std::error_code ec;
    अगर (fs::create_directories(p, ec)) { वापस true; }
    std::error_code ec_stat;
    वापस fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#अगर defined(_WIN32)
    स्थिर अक्षर * base = std::दो_पर्या("LOCALAPPDATA");
    अगर (!base || !*base) { base = std::दो_पर्या("APPDATA"); }
    अगर (!base || !*base) { base = std::दो_पर्या("TEMP"); }
    अगर (!base || !*base) { base = "."; }
    वापस std::string(base) + "\\llama.cpp\\cl-cache";
#या_अगर defined(__APPLE__)
    स्थिर अक्षर * home = std::दो_पर्या("HOME");
    अगर (!home || !*home) { home = "."; }
    वापस std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#अन्यथा
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    स्थिर fs::path tmp_path = fs::temp_directory_path(ec);
    अगर (ec || tmp_path.empty()) { वापस {}; }
    वापस tmp_path.string() + "/llama.cpp/cl-cache";
#पूर्ण_अगर
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
ढाँचा <typename GetInfoFn, typename Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    माप_प्रकार sz = 0;
    अगर (fn(obj, name, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) {
        वापस {};
    }
    std::string s(sz, '\0');
    अगर (fn(obj, name, sz, &s[0], nullptr) != CL_SUCCESS) {
        वापस {};
    }
    अगर (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    वापस s;
}

std::string compute_कुंजी_suffix(cl_device_id device) {
    cl_platform_id platform = nullptr;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, माप(platform), &platform, nullptr);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_VERSION);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_VERSION);    s.push_back('\0');
    अगर (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_VERSION); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_PROGRAM_CACHE_FORMAT_VERSION);
    वापस s;
}

स्थिर uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

bool read_all(स्थिर std::string & path, std::vector<uint8_t> & out) {
    std::अख_प्रवाह f(fs::u8path(path), std::पनप::द्विग);
    अगर (!f) { वापस false; }
    f.ग_स्थान(0, std::पनप::end);
    std::streamsize sz = f.tellg();
    अगर (sz < 0) { वापस false; }
    f.ग_स्थान(0, std::पनप::beg);
    out.resize((माप_प्रकार) sz);
    अगर (sz > 0) { f.पढ़ो((अक्षर *) out.तथ्य(), sz); }
    वापस f.अच्छा() || f.खप();
}

bool write_atomic(स्थिर std::string & path, स्थिर uint8_t * तथ्य, माप_प्रकार len) {
    स्थिर fs::path dst = fs::u8path(path);
    स्थिर fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::बख_प्रवाह f(tmp, std::पनप::द्विग | std::पनप::छाँटो);
        अगर (!f) { वापस false; }
        f.ग_लिखो((स्थिर अक्षर *) तथ्य, (std::streamsize) len);
        अगर (!f.अच्छा()) {
            std::error_code ec_rm;
            fs::हटाओ(tmp, ec_rm);
            वापस false;
        }
    }

    std::error_code ec;
    fs::नाम(tmp, dst, ec);
    अगर (ec) {
        std::error_code ec_rm;
        fs::हटाओ(tmp, ec_rm);
        वापस false;
    }
    वापस true;
}

}  // namespace

अटल bool cache_debug_enabled() {
    अटल पूर्णांक cached = -1;
    अगर (cached < 0) {
        स्थिर अक्षर * e = std::दो_पर्या("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    वापस cached != 0;
}

अटल std::string opts_preview(स्थिर std::string & opts, माप_प्रकार n = 120) {
    अगर (opts.size() <= n) { वापस opts; }
    वापस opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
अटल पूर्णांक g_cache_hits = 0, g_cache_misses = 0, g_cache_saves = 0;

// Debug trace directly to stderr
अटल व्योम cache_debug_पंक्ति(स्थिर अक्षर * kind, स्थिर std::string & कुंजी,
                             स्थिर अक्षर * source, स्थिर std::string & opts) {
    अगर (!cache_debug_enabled()) { वापस; }
    ख_लिखो(मानक_त्रुटि, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            kind, g_cache_hits, g_cache_misses, g_cache_saves,
            कुंजी.substr(0, 16).c_str(), म_माप(source), opts_preview(opts).c_str());
    ख_साफ(मानक_त्रुटि);
}

cl_program_cache_state cl_program_cache_init(cl_device_id device) {
    cl_program_cache_state st;

    स्थिर अक्षर * env = std::दो_पर्या("GGML_OPENCL_KERNEL_CACHE_DIR");
    अगर (env && (!std::म_भेद(env, "0")    || !std::म_भेद(env, "off")  ||
                !std::म_भेद(env, "none") || !std::म_भेद(env, "disable") ||
                !std::म_भेद(env, "disabled"))) {
        अगर (cache_debug_enabled()) {
            ख_लिखो(मानक_त्रुटि, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            ख_साफ(मानक_त्रुटि);
        }
        वापस st;
    }

    std::string dir;
    अगर (!env || !*env || !std::म_भेद(env, "1") || !std::म_भेद(env, "default")) {
        dir = default_cache_dir();
        अगर (dir.empty()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            वापस st;
        }
    } अन्यथा {
        dir = env;
    }

    अगर (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        वापस st;
    }

    st.dir        = dir;
    st.कुंजी_suffix = compute_कुंजी_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    अगर (cache_debug_enabled()) {
        ख_लिखो(मानक_त्रुटि, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        ख_साफ(मानक_त्रुटि);
    }
    वापस st;
}

cl_program cl_program_cache_try_load(
    स्थिर cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    स्थिर अक्षर *                   source,
    स्थिर std::string &            compile_opts) {

    अगर (state.dir.empty() || !source) { वापस nullptr; }

    स्थिर std::string कुंजी  = compute_कुंजी(state.कुंजी_suffix, source, compile_opts);
    स्थिर std::string path = state.dir + "/" + कुंजी + ".clbin";

    std::vector<uint8_t> file;
    अगर (!read_all(path, file)) {
        ++g_cache_misses;
        cache_debug_पंक्ति("MISS", कुंजी, source, compile_opts);
        वापस nullptr;
    }
    अगर (file.size() < 16 || std::स_भेद(file.तथ्य(), MAGIC, 8) != 0) { वापस nullptr; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    अगर (fmt != CL_PROGRAM_CACHE_FORMAT_VERSION) { वापस nullptr; }

    स्थिर माप_प्रकार hdr_len = 16;
    स्थिर अचिन्हित अक्षर * bin     = file.तथ्य() + hdr_len;
    स्थिर माप_प्रकार          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_program p = cl_Create_Program_With_द्विआधारी(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    अगर (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == nullptr) {
        अगर (p) { clReleaseProgram(p); }
        वापस nullptr;
    }

    err = clBuildProgram(p, 0, nullptr, compile_opts.c_str(), nullptr, nullptr);
    अगर (err != CL_SUCCESS) {
        clReleaseProgram(p);
        वापस nullptr;
    }
    ++g_cache_hits;
    cache_debug_पंक्ति("HIT", कुंजी, source, compile_opts);
    वापस p;
}

व्योम cl_program_cache_try_save(
    स्थिर cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   /*device*/,
    स्थिर अक्षर *                   source,
    स्थिर std::string &            compile_opts) {

    अगर (state.dir.empty() || !program || !source) {
        वापस;
    }

    cl_uint n_dev = 0;
    अगर (clGetProgramInfo(program, CL_PROGRAM_अंक_DEVICES, माप(n_dev), &n_dev, nullptr) != CL_SUCCESS || n_dev == 0) {
        वापस;
    }

    std::vector<माप_प्रकार> sizes(n_dev);
    अगर (clGetProgramInfo(program, CL_PROGRAM_द्विआधारी_SIZES, माप(माप_प्रकार) * n_dev, sizes.तथ्य(), nullptr) != CL_SUCCESS) {
        वापस;
    }
    अगर (sizes.empty() || sizes[0] == 0) {
        वापस;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<अचिन्हित अक्षर *>      bin_ptrs(n_dev);
    क्रम (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].तथ्य();
    }
    अगर (clGetProgramInfo(program, CL_PROGRAM_BINARIES, माप(अचिन्हित अक्षर *) * n_dev, bin_ptrs.तथ्य(), nullptr) != CL_SUCCESS) {
        वापस;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    स्थिर std::vector<uint8_t> & bin = binaries[0];

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

    स्थिर std::string कुंजी  = compute_कुंजी(state.कुंजी_suffix, source, compile_opts);
    स्थिर std::string path = state.dir + "/" + कुंजी + ".clbin";
    अगर (!write_atomic(path, file.तथ्य(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } अन्यथा {
        ++g_cache_saves;
        cache_debug_पंक्ति("SAVE", कुंजी, source, compile_opts);
    }
}
