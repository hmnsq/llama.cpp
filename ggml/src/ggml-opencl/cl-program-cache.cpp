// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#परिभाषय CL_TARGET_OPENCL_संस्करणम् GGML_OPENCL_TARGET_संस्करणम्
#परिभाषय CL_USE_DEPRECATED_OPENCL_1_2_APIS

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
#  ifndef WIN32_LEAN_AND_माध्यम्
#    define WIN32_LEAN_AND_माध्यम्
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
#यदि_अन्तः

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
    size_t   buf_len;
};

ध्रुव uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

अन्तःस्थ uint32_t rotr32(uint32_t x, अचिह्नित n) { प्रत्यागम (x >> n) | (x << (32 - n)); }

शून्य sha256_compress(uint32_t state[8], ध्रुव uint8_t खण्डम्[64]) {
    uint32_t w[64];
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)खण्डम्[i*4    ] << 24) |
               ((uint32_t)खण्डम्[i*4 + 1] << 16) |
               ((uint32_t)खण्डम्[i*4 + 2] <<  8) |
               ((uint32_t)खण्डम्[i*4 + 3]      );
    }
    प्रति (पूर्णाङ्क i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    प्रति (पूर्णाङ्क i = 0; i < 64; ++i) {
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

शून्य sha256_update(sha256_ctx & c, ध्रुव शून्य * दत्त, size_t len) {
    ध्रुव uint8_t * p = (ध्रुव uint8_t *) दत्त;
    c.bitlen += (uint64_t) len * 8;
    यदि (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        यदि (n > len) { n = len; }
        स्मृतिं_प्रतिलिख(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        यदि (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    यावत् (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    यदि (len > 0) {
        स्मृतिं_प्रतिलिख(c.buf, p, len);
        c.buf_len = len;
    }
}

शून्य sha256_final(sha256_ctx & c, uint8_t बहिर्गमनम्[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    यदि (c.buf_len > 56) {
        यावत् (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    यावत् (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    प्रति (पूर्णाङ्क i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
        बहिर्गमनम्[i*4    ] = (uint8_t) (c.state[i] >> 24);
        बहिर्गमनम्[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        बहिर्गमनम्[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        बहिर्गमनम्[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(ध्रुव uint8_t digest[32]) {
    स्थिर ध्रुव वर्ण hex[] = "0123456789abcdef";
    std::string s(64, '0');
    प्रति (पूर्णाङ्क i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    प्रत्यागम s;
}

std::string compute_कुञ्चिका(ध्रुव std::string & कुञ्चिका_suffix,
                        ध्रुव वर्ण *        source,
                        ध्रुव std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    स्थिर ध्रुव uint8_t sep = 0;
    sha256_update(c, source,             वाक्यदैर्घ्यम्(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.दत्त(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, कुञ्चिका_suffix.दत्त(),  कुञ्चिका_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    प्रत्यागम sha256_hex(digest);
}

तार्किक make_dir_recursive(ध्रुव std::string & path) {
    यदि (path.रिक्तानि()) { प्रत्यागम असत्य; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    ध्रुव fs::path p = fs::u8path(path);
    std::error_code ec;
    यदि (fs::create_directories(p, ec)) { प्रत्यागम सत्य; }
    std::error_code ec_stat;
    प्रत्यागम fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#यदि defined(_WIN32)
    ध्रुव वर्ण * आधारः = std::getenv("LOCALAPPDATA");
    यदि (!आधारः || !*आधारः) { आधारः = std::getenv("APPDATA"); }
    यदि (!आधारः || !*आधारः) { आधारः = std::getenv("TEMP"); }
    यदि (!आधारः || !*आधारः) { आधारः = "."; }
    प्रत्यागम std::string(आधारः) + "\\llama.cpp\\cl-cache";
#नोचेद्यदि defined(__APPLE__)
    ध्रुव वर्ण * home = std::getenv("HOME");
    यदि (!home || !*home) { home = "."; }
    प्रत्यागम std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#अन्यथा
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    ध्रुव fs::path tmp_मार्गः = fs::temp_directory_path(ec);
    यदि (ec || tmp_मार्गः.रिक्तानि()) { प्रत्यागम {}; }
    प्रत्यागम tmp_मार्गः.string() + "/llama.cpp/cl-cache";
#यदि_अन्तः
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
प्रतिरूपम् <प्रकारनाम GetInfoFn, प्रकारनाम Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    यदि (fn(obj, name, 0, शून्यसूचकः, &sz) != CL_SUCCESS || sz == 0) {
        प्रत्यागम {};
    }
    std::string s(sz, '\0');
    यदि (fn(obj, name, sz, &s[0], शून्यसूचकः) != CL_SUCCESS) {
        प्रत्यागम {};
    }
    यदि (!s.रिक्तानि() && s.back() == '\0') {
        s.pop_back();
    }
    प्रत्यागम s;
}

std::string compute_कुञ्चिका_suffix(cl_device_id device) {
    cl_platform_id platform = शून्यसूचकः;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, परिमाण(platform), &platform, शून्यसूचकः);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_संस्करणम्);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_संस्करणम्);    s.push_back('\0');
    यदि (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_संस्करणम्); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_क्रमादेशः_CACHE_FORMAT_संस्करणम्);
    प्रत्यागम s;
}

ध्रुव uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

तार्किक read_all(ध्रुव std::string & path, std::vector<uint8_t> & बहिर्गमनम्) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    यदि (!f) { प्रत्यागम असत्य; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    यदि (sz < 0) { प्रत्यागम असत्य; }
    f.seekg(0, std::ios::beg);
    बहिर्गमनम्.resize((size_t) sz);
    यदि (sz > 0) { f.पठ((वर्ण *) बहिर्गमनम्.दत्त(), sz); }
    प्रत्यागम f.good() || f.eof();
}

तार्किक write_atomic(ध्रुव std::string & path, ध्रुव uint8_t * दत्त, size_t len) {
    ध्रुव fs::path dst = fs::u8path(path);
    ध्रुव fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        यदि (!f) { प्रत्यागम असत्य; }
        f.लिख((ध्रुव वर्ण *) दत्त, (std::streamsize) len);
        यदि (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            प्रत्यागम असत्य;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    यदि (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

}  // namespace

स्थिर तार्किक cache_debug_enabled() {
    स्थिर पूर्णाङ्क cached = -1;
    यदि (cached < 0) {
        ध्रुव वर्ण * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    प्रत्यागम cached != 0;
}

स्थिर std::string opts_preview(ध्रुव std::string & opts, size_t n = 120) {
    यदि (opts.size() <= n) { प्रत्यागम opts; }
    प्रत्यागम opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
स्थिर पूर्णाङ्क g_cache_लब्धानि = 0, g_cache_अलब्धानि = 0, g_cache_saves = 0;

// Debug trace directly to stderr
स्थिर शून्य cache_debug_पङ्क्तिः(ध्रुव वर्ण * विधा, ध्रुव std::string & कुञ्चिका,
                             ध्रुव वर्ण * source, ध्रुव std::string & opts) {
    यदि (!cache_debug_enabled()) { प्रत्यागम; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            विधा, g_cache_लब्धानि, g_cache_अलब्धानि, g_cache_saves,
            कुञ्चिका.substr(0, 16).c_str(), वाक्यदैर्घ्यम्(source), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_क्रमादेशः_cache_state cl_क्रमादेशः_cache_init(cl_device_id device) {
    cl_क्रमादेशः_cache_state st;

    ध्रुव वर्ण * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    यदि (env && (!std::वाक्यं_तुलय(env, "0")    || !std::वाक्यं_तुलय(env, "off")  ||
                !std::वाक्यं_तुलय(env, "none") || !std::वाक्यं_तुलय(env, "disable") ||
                !std::वाक्यं_तुलय(env, "disabled"))) {
        यदि (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        प्रत्यागम st;
    }

    std::string dir;
    यदि (!env || !*env || !std::वाक्यं_तुलय(env, "1") || !std::वाक्यं_तुलय(env, "default")) {
        dir = default_cache_dir();
        यदि (dir.रिक्तानि()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            प्रत्यागम st;
        }
    } अन्यथा {
        dir = env;
    }

    यदि (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        प्रत्यागम st;
    }

    st.dir        = dir;
    st.कुञ्चिका_suffix = compute_कुञ्चिका_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    यदि (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    प्रत्यागम st;
}

cl_क्रमादेशः cl_क्रमादेशः_cache_try_आरोपय(
    ध्रुव cl_क्रमादेशः_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    ध्रुव वर्ण *                   source,
    ध्रुव std::string &            compile_opts) {

    यदि (state.dir.रिक्तानि() || !source) { प्रत्यागम शून्यसूचकः; }

    ध्रुव std::string कुञ्चिका  = compute_कुञ्चिका(state.कुञ्चिका_suffix, source, compile_opts);
    ध्रुव std::string path = state.dir + "/" + कुञ्चिका + ".clbin";

    std::vector<uint8_t> file;
    यदि (!read_all(path, file)) {
        ++g_cache_अलब्धानि;
        cache_debug_पङ्क्तिः("MISS", कुञ्चिका, source, compile_opts);
        प्रत्यागम शून्यसूचकः;
    }
    यदि (file.size() < 16 || std::memcmp(file.दत्त(), MAGIC, 8) != 0) { प्रत्यागम शून्यसूचकः; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    यदि (fmt != CL_क्रमादेशः_CACHE_FORMAT_संस्करणम्) { प्रत्यागम शून्यसूचकः; }

    ध्रुव size_t hdr_len = 16;
    ध्रुव अचिह्नित वर्ण * bin     = file.दत्त() + hdr_len;
    ध्रुव size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_क्रमादेशः p = cl_Create_क्रमादेशः_With_द्विमान(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    यदि (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == शून्यसूचकः) {
        यदि (p) { cl_Release_क्रमादेशः(p); }
        प्रत्यागम शून्यसूचकः;
    }

    err = cl_Build_क्रमादेशः(p, 0, शून्यसूचकः, compile_opts.c_str(), शून्यसूचकः, शून्यसूचकः);
    यदि (err != CL_SUCCESS) {
        cl_Release_क्रमादेशः(p);
        प्रत्यागम शून्यसूचकः;
    }
    ++g_cache_लब्धानि;
    cache_debug_पङ्क्तिः("HIT", कुञ्चिका, source, compile_opts);
    प्रत्यागम p;
}

शून्य cl_क्रमादेशः_cache_try_save(
    ध्रुव cl_क्रमादेशः_cache_state & state,
    cl_क्रमादेशः                     क्रमादेशः,
    cl_device_id                   /*device*/,
    ध्रुव वर्ण *                   source,
    ध्रुव std::string &            compile_opts) {

    यदि (state.dir.रिक्तानि() || !क्रमादेशः || !source) {
        प्रत्यागम;
    }

    cl_uint n_dev = 0;
    यदि (cl_Get_क्रमादेशः_Info(क्रमादेशः, CL_क्रमादेशः_अङ्कः_DEVICES, परिमाण(n_dev), &n_dev, शून्यसूचकः) != CL_SUCCESS || n_dev == 0) {
        प्रत्यागम;
    }

    std::vector<size_t> sizes(n_dev);
    यदि (cl_Get_क्रमादेशः_Info(क्रमादेशः, CL_क्रमादेशः_द्विमान_SIZES, परिमाण(size_t) * n_dev, sizes.दत्त(), शून्यसूचकः) != CL_SUCCESS) {
        प्रत्यागम;
    }
    यदि (sizes.रिक्तानि() || sizes[0] == 0) {
        प्रत्यागम;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<अचिह्नित वर्ण *>      bin_ptrs(n_dev);
    प्रति (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].दत्त();
    }
    यदि (cl_Get_क्रमादेशः_Info(क्रमादेशः, CL_क्रमादेशः_BINARIES, परिमाण(अचिह्नित वर्ण *) * n_dev, bin_ptrs.दत्त(), शून्यसूचकः) != CL_SUCCESS) {
        प्रत्यागम;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    ध्रुव std::vector<uint8_t> & bin = binaries[0];

    std::vector<uint8_t> file;
    file.reserve(16 + bin.size());
    file.insert(file.end(), MAGIC, MAGIC + 8);
    uint32_t fmt = CL_क्रमादेशः_CACHE_FORMAT_संस्करणम्;
    file.push_back((uint8_t)  (fmt        & 0xff));
    file.push_back((uint8_t) ((fmt >>  8) & 0xff));
    file.push_back((uint8_t) ((fmt >> 16) & 0xff));
    file.push_back((uint8_t) ((fmt >> 24) & 0xff));
    file.push_back(0); file.push_back(0); file.push_back(0); file.push_back(0); // reserved
    file.insert(file.end(), bin.begin(), bin.end());

    ध्रुव std::string कुञ्चिका  = compute_कुञ्चिका(state.कुञ्चिका_suffix, source, compile_opts);
    ध्रुव std::string path = state.dir + "/" + कुञ्चिका + ".clbin";
    यदि (!write_atomic(path, file.दत्त(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } अन्यथा {
        ++g_cache_saves;
        cache_debug_पङ्क्तिः("SAVE", कुञ्चिका, source, compile_opts);
    }
}
