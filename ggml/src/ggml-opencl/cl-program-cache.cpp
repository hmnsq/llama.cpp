// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#నిర్వచించు CL_లక్ష్యం_OPENCL_సంస్కరణ GGML_OPENCL_లక్ష్యం_సంస్కరణ
#నిర్వచించు CL_USE_DEPRECATED_OPENCL_1_2_APIS

#అన్తర్భావ "cl-program-cache.h"

#అన్తర్భావ "ggml-impl.h"  // GGML_LOG_INFO / WARN

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <filesystem>
#అన్తర్భావ <fstream>
#అన్తర్భావ <system_error>
#అన్తర్భావ <vector>

#ఒకవేళ defined(_WIN32)
#  ifndef WIN32_LEAN_AND_సగటు
#    define WIN32_LEAN_AND_సగటు
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((పూర్ణాంకం) GetCurrentProcessId())
#లేకపోతే
#  include <unistd.h>
#  define ggml_getpid()    ((పూర్ణాంకం) getpid())
#అయితే_ముగింపు

నామాకాశ fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

నామాకాశ {

నిర్మాణం sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buf_len;
};

స్థిరరాశి uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

అంతఃవరుస uint32_t rotr32(uint32_t x, గుర్తులేని n) { తిరిగి (x >> n) | (x << (32 - n)); }

శూన్య sha256_compress(uint32_t state[8], స్థిరరాశి uint8_t ఖండం[64]) {
    uint32_t w[64];
    కొరకు (పూర్ణాంకం i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)ఖండం[i*4    ] << 24) |
               ((uint32_t)ఖండం[i*4 + 1] << 16) |
               ((uint32_t)ఖండం[i*4 + 2] <<  8) |
               ((uint32_t)ఖండం[i*4 + 3]      );
    }
    కొరకు (పూర్ణాంకం i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    కొరకు (పూర్ణాంకం i = 0; i < 64; ++i) {
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

శూన్య sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

శూన్య sha256_update(sha256_ctx & c, స్థిరరాశి శూన్య * దత్తాంశం, size_t len) {
    స్థిరరాశి uint8_t * p = (స్థిరరాశి uint8_t *) దత్తాంశం;
    c.bitlen += (uint64_t) len * 8;
    ఒకవేళ (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        ఒకవేళ (n > len) { n = len; }
        జ్ఞాపకం_నకలు(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        ఒకవేళ (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    అయితే (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    ఒకవేళ (len > 0) {
        జ్ఞాపకం_నకలు(c.buf, p, len);
        c.buf_len = len;
    }
}

శూన్య sha256_final(sha256_ctx & c, uint8_t బహిర్గతం[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    ఒకవేళ (c.buf_len > 56) {
        అయితే (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    అయితే (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    కొరకు (పూర్ణాంకం i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    కొరకు (పూర్ణాంకం i = 0; i < 8; ++i) {
        బహిర్గతం[i*4    ] = (uint8_t) (c.state[i] >> 24);
        బహిర్గతం[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        బహిర్గతం[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        బహిర్గతం[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(స్థిరరాశి uint8_t digest[32]) {
    స్థిర స్థిరరాశి అక్షరం hex[] = "0123456789abcdef";
    std::string s(64, '0');
    కొరకు (పూర్ణాంకం i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    తిరిగి s;
}

std::string compute_కీలకం(స్థిరరాశి std::string & కీలకం_suffix,
                        స్థిరరాశి అక్షరం *        ఆధారం,
                        స్థిరరాశి std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    స్థిర స్థిరరాశి uint8_t sep = 0;
    sha256_update(c, ఆధారం,             అక్షరశ్రేణి_పొడవు(ఆధారం));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.దత్తాంశం(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, కీలకం_suffix.దత్తాంశం(),  కీలకం_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    తిరిగి sha256_hex(digest);
}

తార్కిక make_dir_recursive(స్థిరరాశి std::string & path) {
    ఒకవేళ (path.ఖాళీలు()) { తిరిగి అబద్ధం; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    స్థిరరాశి fs::path p = fs::u8path(path);
    std::error_code ec;
    ఒకవేళ (fs::create_directories(p, ec)) { తిరిగి నిజం; }
    std::error_code ec_stat;
    తిరిగి fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#ఒకవేళ defined(_WIN32)
    స్థిరరాశి అక్షరం * పునాది = std::getenv("LOCALAPPDATA");
    ఒకవేళ (!పునాది || !*పునాది) { పునాది = std::getenv("APPDATA"); }
    ఒకవేళ (!పునాది || !*పునాది) { పునాది = std::getenv("TEMP"); }
    ఒకవేళ (!పునాది || !*పునాది) { పునాది = "."; }
    తిరిగి std::string(పునాది) + "\\llama.cpp\\cl-cache";
#లేదావేళ defined(__APPLE__)
    స్థిరరాశి అక్షరం * home = std::getenv("HOME");
    ఒకవేళ (!home || !*home) { home = "."; }
    తిరిగి std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#లేకపోతే
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    స్థిరరాశి fs::path tmp_మార్గం = fs::temp_directory_path(ec);
    ఒకవేళ (ec || tmp_మార్గం.ఖాళీలు()) { తిరిగి {}; }
    తిరిగి tmp_మార్గం.string() + "/llama.cpp/cl-cache";
#అయితే_ముగింపు
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
మూస <రకం_పేరు GetInfoFn, రకం_పేరు Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    ఒకవేళ (fn(obj, name, 0, శూన్యసూచిక, &sz) != CL_SUCCESS || sz == 0) {
        తిరిగి {};
    }
    std::string s(sz, '\0');
    ఒకవేళ (fn(obj, name, sz, &s[0], శూన్యసూచిక) != CL_SUCCESS) {
        తిరిగి {};
    }
    ఒకవేళ (!s.ఖాళీలు() && s.back() == '\0') {
        s.pop_back();
    }
    తిరిగి s;
}

std::string compute_కీలకం_suffix(cl_device_id device) {
    cl_platform_id platform = శూన్యసూచిక;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, పరిమాణం(platform), &platform, శూన్యసూచిక);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_సంస్కరణ);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_సంస్కరణ);    s.push_back('\0');
    ఒకవేళ (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_సంస్కరణ); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_కార్యక్రమం_CACHE_FORMAT_సంస్కరణ);
    తిరిగి s;
}

స్థిరరాశి uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

తార్కిక read_all(స్థిరరాశి std::string & path, std::vector<uint8_t> & బహిర్గతం) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    ఒకవేళ (!f) { తిరిగి అబద్ధం; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    ఒకవేళ (sz < 0) { తిరిగి అబద్ధం; }
    f.seekg(0, std::ios::beg);
    బహిర్గతం.resize((size_t) sz);
    ఒకవేళ (sz > 0) { f.పఠ((అక్షరం *) బహిర్గతం.దత్తాంశం(), sz); }
    తిరిగి f.good() || f.eof();
}

తార్కిక write_atomic(స్థిరరాశి std::string & path, స్థిరరాశి uint8_t * దత్తాంశం, size_t len) {
    స్థిరరాశి fs::path dst = fs::u8path(path);
    స్థిరరాశి fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        ఒకవేళ (!f) { తిరిగి అబద్ధం; }
        f.లిఖ((స్థిరరాశి అక్షరం *) దత్తాంశం, (std::streamsize) len);
        ఒకవేళ (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            తిరిగి అబద్ధం;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    ఒకవేళ (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

}  // namespace

స్థిర తార్కిక cache_debug_enabled() {
    స్థిర పూర్ణాంకం cached = -1;
    ఒకవేళ (cached < 0) {
        స్థిరరాశి అక్షరం * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    తిరిగి cached != 0;
}

స్థిర std::string opts_preview(స్థిరరాశి std::string & opts, size_t n = 120) {
    ఒకవేళ (opts.size() <= n) { తిరిగి opts; }
    తిరిగి opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
స్థిర పూర్ణాంకం g_cache_తగిలినవి = 0, g_cache_తప్పినవి = 0, g_cache_saves = 0;

// Debug trace directly to stderr
స్థిర శూన్య cache_debug_పంక్తి(స్థిరరాశి అక్షరం * రకం_2, స్థిరరాశి std::string & కీలకం,
                             స్థిరరాశి అక్షరం * ఆధారం, స్థిరరాశి std::string & opts) {
    ఒకవేళ (!cache_debug_enabled()) { తిరిగి; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            రకం_2, g_cache_తగిలినవి, g_cache_తప్పినవి, g_cache_saves,
            కీలకం.substr(0, 16).c_str(), అక్షరశ్రేణి_పొడవు(ఆధారం), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_కార్యక్రమం_cache_state cl_కార్యక్రమం_cache_init(cl_device_id device) {
    cl_కార్యక్రమం_cache_state st;

    స్థిరరాశి అక్షరం * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    ఒకవేళ (env && (!std::strcmp(env, "0")    || !std::strcmp(env, "off")  ||
                !std::strcmp(env, "none") || !std::strcmp(env, "disable") ||
                !std::strcmp(env, "disabled"))) {
        ఒకవేళ (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        తిరిగి st;
    }

    std::string dir;
    ఒకవేళ (!env || !*env || !std::strcmp(env, "1") || !std::strcmp(env, "default")) {
        dir = default_cache_dir();
        ఒకవేళ (dir.ఖాళీలు()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            తిరిగి st;
        }
    } లేకపోతే {
        dir = env;
    }

    ఒకవేళ (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        తిరిగి st;
    }

    st.dir        = dir;
    st.కీలకం_suffix = compute_కీలకం_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    ఒకవేళ (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    తిరిగి st;
}

cl_కార్యక్రమం cl_కార్యక్రమం_cache_try_ఎక్కించు(
    స్థిరరాశి cl_కార్యక్రమం_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    స్థిరరాశి అక్షరం *                   ఆధారం,
    స్థిరరాశి std::string &            compile_opts) {

    ఒకవేళ (state.dir.ఖాళీలు() || !ఆధారం) { తిరిగి శూన్యసూచిక; }

    స్థిరరాశి std::string కీలకం  = compute_కీలకం(state.కీలకం_suffix, ఆధారం, compile_opts);
    స్థిరరాశి std::string path = state.dir + "/" + కీలకం + ".clbin";

    std::vector<uint8_t> file;
    ఒకవేళ (!read_all(path, file)) {
        ++g_cache_తప్పినవి;
        cache_debug_పంక్తి("MISS", కీలకం, ఆధారం, compile_opts);
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (file.size() < 16 || std::memcmp(file.దత్తాంశం(), MAGIC, 8) != 0) { తిరిగి శూన్యసూచిక; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    ఒకవేళ (fmt != CL_కార్యక్రమం_CACHE_FORMAT_సంస్కరణ) { తిరిగి శూన్యసూచిక; }

    స్థిరరాశి size_t hdr_len = 16;
    స్థిరరాశి గుర్తులేని అక్షరం * bin     = file.దత్తాంశం() + hdr_len;
    స్థిరరాశి size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_కార్యక్రమం p = cl_Create_కార్యక్రమం_With_ద్విమాన(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    ఒకవేళ (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == శూన్యసూచిక) {
        ఒకవేళ (p) { cl_Release_కార్యక్రమం(p); }
        తిరిగి శూన్యసూచిక;
    }

    err = cl_Build_కార్యక్రమం(p, 0, శూన్యసూచిక, compile_opts.c_str(), శూన్యసూచిక, శూన్యసూచిక);
    ఒకవేళ (err != CL_SUCCESS) {
        cl_Release_కార్యక్రమం(p);
        తిరిగి శూన్యసూచిక;
    }
    ++g_cache_తగిలినవి;
    cache_debug_పంక్తి("HIT", కీలకం, ఆధారం, compile_opts);
    తిరిగి p;
}

శూన్య cl_కార్యక్రమం_cache_try_save(
    స్థిరరాశి cl_కార్యక్రమం_cache_state & state,
    cl_కార్యక్రమం                     కార్యక్రమం,
    cl_device_id                   /*device*/,
    స్థిరరాశి అక్షరం *                   ఆధారం,
    స్థిరరాశి std::string &            compile_opts) {

    ఒకవేళ (state.dir.ఖాళీలు() || !కార్యక్రమం || !ఆధారం) {
        తిరిగి;
    }

    cl_uint n_dev = 0;
    ఒకవేళ (cl_Get_కార్యక్రమం_Info(కార్యక్రమం, CL_కార్యక్రమం_అంకె_DEVICES, పరిమాణం(n_dev), &n_dev, శూన్యసూచిక) != CL_SUCCESS || n_dev == 0) {
        తిరిగి;
    }

    std::vector<size_t> sizes(n_dev);
    ఒకవేళ (cl_Get_కార్యక్రమం_Info(కార్యక్రమం, CL_కార్యక్రమం_ద్విమాన_SIZES, పరిమాణం(size_t) * n_dev, sizes.దత్తాంశం(), శూన్యసూచిక) != CL_SUCCESS) {
        తిరిగి;
    }
    ఒకవేళ (sizes.ఖాళీలు() || sizes[0] == 0) {
        తిరిగి;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<గుర్తులేని అక్షరం *>      bin_ptrs(n_dev);
    కొరకు (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].దత్తాంశం();
    }
    ఒకవేళ (cl_Get_కార్యక్రమం_Info(కార్యక్రమం, CL_కార్యక్రమం_BINARIES, పరిమాణం(గుర్తులేని అక్షరం *) * n_dev, bin_ptrs.దత్తాంశం(), శూన్యసూచిక) != CL_SUCCESS) {
        తిరిగి;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    స్థిరరాశి std::vector<uint8_t> & bin = binaries[0];

    std::vector<uint8_t> file;
    file.reserve(16 + bin.size());
    file.insert(file.end(), MAGIC, MAGIC + 8);
    uint32_t fmt = CL_కార్యక్రమం_CACHE_FORMAT_సంస్కరణ;
    file.push_back((uint8_t)  (fmt        & 0xff));
    file.push_back((uint8_t) ((fmt >>  8) & 0xff));
    file.push_back((uint8_t) ((fmt >> 16) & 0xff));
    file.push_back((uint8_t) ((fmt >> 24) & 0xff));
    file.push_back(0); file.push_back(0); file.push_back(0); file.push_back(0); // reserved
    file.insert(file.end(), bin.begin(), bin.end());

    స్థిరరాశి std::string కీలకం  = compute_కీలకం(state.కీలకం_suffix, ఆధారం, compile_opts);
    స్థిరరాశి std::string path = state.dir + "/" + కీలకం + ".clbin";
    ఒకవేళ (!write_atomic(path, file.దత్తాంశం(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } లేకపోతే {
        ++g_cache_saves;
        cache_debug_పంక్తి("SAVE", కీలకం, ఆధారం, compile_opts);
    }
}
