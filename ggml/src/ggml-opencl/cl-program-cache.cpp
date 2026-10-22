// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#अन्तर्भाव "cl-program-cache.h"

#अन्तर्भाव "ggml-impl.h"  // GGML_LOG_INFO / WARN

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <filesystem>
#अन्तर्भाव <fstream>
#अन्तर्भाव <system_error>
#अन्तर्भाव <vector>

#यदि defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((पूर्णाङ्क) GetCurrentProcessId())
#अन्यथा
#  include <unistd.h>
#  define ggml_getpid()    ((पूर्णाङ्क) getpid())
#endif

नामाकाश fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

नामाकाश {

संरचना sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    माप_प्रकार   buf_len;
};

अचर uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

अन्तःस्थ uint32_t rotr32(uint32_t x, अचिन्हित n) { फर्काऊ (x >> n) | (x << (32 - n)); }

शून्य sha256_compress(uint32_t state[8], अचर uint8_t block[64]) {
    uint32_t w[64];
    लागि (पूर्णाङ्क i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4    ] << 24) |
               ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] <<  8) |
               ((uint32_t)block[i*4 + 3]      );
    }
    लागि (पूर्णाङ्क i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    लागि (पूर्णाङ्क i = 0; i < 64; ++i) {
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

शून्य sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

शून्य sha256_update(sha256_ctx & c, अचर शून्य * data, माप_प्रकार len) {
    अचर uint8_t * p = (अचर uint8_t *) data;
    c.bitlen += (uint64_t) len * 8;
    यदि (c.buf_len > 0) {
        माप_प्रकार n = 64 - c.buf_len;
        यदि (n > len) { n = len; }
        स_नकल(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        यदि (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    जबसम्म (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    यदि (len > 0) {
        स_नकल(c.buf, p, len);
        c.buf_len = len;
    }
}

शून्य sha256_final(sha256_ctx & c, uint8_t out[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    यदि (c.buf_len > 56) {
        जबसम्म (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    जबसम्म (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    लागि (पूर्णाङ्क i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    लागि (पूर्णाङ्क i = 0; i < 8; ++i) {
        out[i*4    ] = (uint8_t) (c.state[i] >> 24);
        out[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        out[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        out[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(अचर uint8_t digest[32]) {
    स्थिर अचर वर्ण hex[] = "0123456789abcdef";
    std::string s(64, '0');
    लागि (पूर्णाङ्क i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    फर्काऊ s;
}

std::string compute_साँचो(अचर std::string & साँचो_suffix,
                        अचर वर्ण *        source,
                        अचर std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    स्थिर अचर uint8_t sep = 0;
    sha256_update(c, source,             म_माप(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.data(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, साँचो_suffix.data(),  साँचो_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    फर्काऊ sha256_hex(digest);
}

तार्किक make_dir_recursive(अचर std::string & path) {
    यदि (path.empty()) { फर्काऊ असत्य; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    अचर fs::path p = fs::u8path(path);
    std::error_code ec;
    यदि (fs::create_directories(p, ec)) { फर्काऊ सत्य; }
    std::error_code ec_stat;
    फर्काऊ fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#यदि defined(_WIN32)
    अचर वर्ण * base = std::दो_पर्या("LOCALAPPDATA");
    यदि (!base || !*base) { base = std::दो_पर्या("APPDATA"); }
    यदि (!base || !*base) { base = std::दो_पर्या("TEMP"); }
    यदि (!base || !*base) { base = "."; }
    फर्काऊ std::string(base) + "\\llama.cpp\\cl-cache";
#अथवायदि defined(__APPLE__)
    अचर वर्ण * home = std::दो_पर्या("HOME");
    यदि (!home || !*home) { home = "."; }
    फर्काऊ std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#अन्यथा
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    अचर fs::path tmp_path = fs::temp_directory_path(ec);
    यदि (ec || tmp_path.empty()) { फर्काऊ {}; }
    फर्काऊ tmp_path.string() + "/llama.cpp/cl-cache";
#endif
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
ढाँचा <typename GetInfoFn, typename Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    माप_प्रकार sz = 0;
    यदि (fn(obj, name, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) {
        फर्काऊ {};
    }
    std::string s(sz, '\0');
    यदि (fn(obj, name, sz, &s[0], nullptr) != CL_SUCCESS) {
        फर्काऊ {};
    }
    यदि (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    फर्काऊ s;
}

std::string compute_साँचो_suffix(cl_device_id device) {
    cl_platform_id platform = nullptr;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, परिमाण(platform), &platform, nullptr);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_VERSION);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_VERSION);    s.push_back('\0');
    यदि (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_VERSION); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_PROGRAM_CACHE_FORMAT_VERSION);
    फर्काऊ s;
}

अचर uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

तार्किक read_all(अचर std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    यदि (!f) { फर्काऊ असत्य; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    यदि (sz < 0) { फर्काऊ असत्य; }
    f.seekg(0, std::ios::beg);
    out.resize((माप_प्रकार) sz);
    यदि (sz > 0) { f.पठ((वर्ण *) out.data(), sz); }
    फर्काऊ f.good() || f.eof();
}

तार्किक write_atomic(अचर std::string & path, अचर uint8_t * data, माप_प्रकार len) {
    अचर fs::path dst = fs::u8path(path);
    अचर fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        यदि (!f) { फर्काऊ असत्य; }
        f.लिख((अचर वर्ण *) data, (std::streamsize) len);
        यदि (!f.good()) {
            std::error_code ec_rm;
            fs::हटाओ(tmp, ec_rm);
            फर्काऊ असत्य;
        }
    }

    std::error_code ec;
    fs::नाम(tmp, dst, ec);
    यदि (ec) {
        std::error_code ec_rm;
        fs::हटाओ(tmp, ec_rm);
        फर्काऊ असत्य;
    }
    फर्काऊ सत्य;
}

}  // namespace

स्थिर तार्किक cache_debug_enabled() {
    स्थिर पूर्णाङ्क cached = -1;
    यदि (cached < 0) {
        अचर वर्ण * e = std::दो_पर्या("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    फर्काऊ cached != 0;
}

स्थिर std::string opts_preview(अचर std::string & opts, माप_प्रकार n = 120) {
    यदि (opts.size() <= n) { फर्काऊ opts; }
    फर्काऊ opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
स्थिर पूर्णाङ्क g_cache_hits = 0, g_cache_misses = 0, g_cache_saves = 0;

// Debug trace directly to stderr
स्थिर शून्य cache_debug_पङ्क्ति(अचर वर्ण * kind, अचर std::string & साँचो,
                             अचर वर्ण * source, अचर std::string & opts) {
    यदि (!cache_debug_enabled()) { फर्काऊ; }
    ख_लिखो(मानक_त्रुटि, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            kind, g_cache_hits, g_cache_misses, g_cache_saves,
            साँचो.substr(0, 16).c_str(), म_माप(source), opts_preview(opts).c_str());
    ख_साफ(मानक_त्रुटि);
}

cl_program_cache_state cl_program_cache_init(cl_device_id device) {
    cl_program_cache_state st;

    अचर वर्ण * env = std::दो_पर्या("GGML_OPENCL_KERNEL_CACHE_DIR");
    यदि (env && (!std::म_भेद(env, "0")    || !std::म_भेद(env, "off")  ||
                !std::म_भेद(env, "none") || !std::म_भेद(env, "disable") ||
                !std::म_भेद(env, "disabled"))) {
        यदि (cache_debug_enabled()) {
            ख_लिखो(मानक_त्रुटि, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            ख_साफ(मानक_त्रुटि);
        }
        फर्काऊ st;
    }

    std::string dir;
    यदि (!env || !*env || !std::म_भेद(env, "1") || !std::म_भेद(env, "default")) {
        dir = default_cache_dir();
        यदि (dir.empty()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            फर्काऊ st;
        }
    } अन्यथा {
        dir = env;
    }

    यदि (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        फर्काऊ st;
    }

    st.dir        = dir;
    st.साँचो_suffix = compute_साँचो_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    यदि (cache_debug_enabled()) {
        ख_लिखो(मानक_त्रुटि, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        ख_साफ(मानक_त्रुटि);
    }
    फर्काऊ st;
}

cl_program cl_program_cache_try_load(
    अचर cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    अचर वर्ण *                   source,
    अचर std::string &            compile_opts) {

    यदि (state.dir.empty() || !source) { फर्काऊ nullptr; }

    अचर std::string साँचो  = compute_साँचो(state.साँचो_suffix, source, compile_opts);
    अचर std::string path = state.dir + "/" + साँचो + ".clbin";

    std::vector<uint8_t> file;
    यदि (!read_all(path, file)) {
        ++g_cache_misses;
        cache_debug_पङ्क्ति("MISS", साँचो, source, compile_opts);
        फर्काऊ nullptr;
    }
    यदि (file.size() < 16 || std::स_भेद(file.data(), MAGIC, 8) != 0) { फर्काऊ nullptr; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    यदि (fmt != CL_PROGRAM_CACHE_FORMAT_VERSION) { फर्काऊ nullptr; }

    अचर माप_प्रकार hdr_len = 16;
    अचर अचिन्हित वर्ण * bin     = file.data() + hdr_len;
    अचर माप_प्रकार          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_program p = cl_Create_Program_With_द्विआधारी(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    यदि (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == nullptr) {
        यदि (p) { clReleaseProgram(p); }
        फर्काऊ nullptr;
    }

    err = clBuildProgram(p, 0, nullptr, compile_opts.c_str(), nullptr, nullptr);
    यदि (err != CL_SUCCESS) {
        clReleaseProgram(p);
        फर्काऊ nullptr;
    }
    ++g_cache_hits;
    cache_debug_पङ्क्ति("HIT", साँचो, source, compile_opts);
    फर्काऊ p;
}

शून्य cl_program_cache_try_save(
    अचर cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   /*device*/,
    अचर वर्ण *                   source,
    अचर std::string &            compile_opts) {

    यदि (state.dir.empty() || !program || !source) {
        फर्काऊ;
    }

    cl_uint n_dev = 0;
    यदि (clGetProgramInfo(program, CL_PROGRAM_अङ्क_DEVICES, परिमाण(n_dev), &n_dev, nullptr) != CL_SUCCESS || n_dev == 0) {
        फर्काऊ;
    }

    std::vector<माप_प्रकार> sizes(n_dev);
    यदि (clGetProgramInfo(program, CL_PROGRAM_द्विआधारी_SIZES, परिमाण(माप_प्रकार) * n_dev, sizes.data(), nullptr) != CL_SUCCESS) {
        फर्काऊ;
    }
    यदि (sizes.empty() || sizes[0] == 0) {
        फर्काऊ;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<अचिन्हित वर्ण *>      bin_ptrs(n_dev);
    लागि (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].data();
    }
    यदि (clGetProgramInfo(program, CL_PROGRAM_BINARIES, परिमाण(अचिन्हित वर्ण *) * n_dev, bin_ptrs.data(), nullptr) != CL_SUCCESS) {
        फर्काऊ;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    अचर std::vector<uint8_t> & bin = binaries[0];

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

    अचर std::string साँचो  = compute_साँचो(state.साँचो_suffix, source, compile_opts);
    अचर std::string path = state.dir + "/" + साँचो + ".clbin";
    यदि (!write_atomic(path, file.data(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } अन्यथा {
        ++g_cache_saves;
        cache_debug_पङ्क्ति("SAVE", साँचो, source, compile_opts);
    }
}
