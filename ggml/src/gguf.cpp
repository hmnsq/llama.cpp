#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "gguf.h"

#అన్తర్భావ <cinttypes>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <cerrno>
#అన్తర్భావ <map>
#అన్తర్భావ <new>
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <string>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <vector>

#నిర్వచించు GGUF_MAX_STRING_నిడివి  (1024*1024*1024)
#నిర్వచించు GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#నిర్వచితమైతే _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#లేకపోతే
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#అయితే_ముగింపు

మూస <రకం_పేరు T>
నిర్మాణం type_గమ్యం_gguf_type;

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<uint8_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_UINT8;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<int8_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_INT8;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<uint16_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_UINT16;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<int16_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_INT16;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<uint32_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_UINT32;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<int32_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_INT32;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<దశాంశం> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_FLOAT32;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<తార్కిక> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_BOOL;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<std::string> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_STRING;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<uint64_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_UINT64;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<int64_t> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_INT64;
};

మూస <>
నిర్మాణం type_గమ్యం_gguf_type<ద్విగుణ> {
    స్థిర constexpr గణన gguf_type value = GGUF_TYPE_FLOAT64;
};

స్థిర స్థిరరాశి std::map<gguf_type, size_t> GGUF_TYPE_కొలత = {
    {GGUF_TYPE_UINT8,   పరిమాణం(uint8_t)},
    {GGUF_TYPE_INT8,    పరిమాణం(int8_t)},
    {GGUF_TYPE_UINT16,  పరిమాణం(uint16_t)},
    {GGUF_TYPE_INT16,   పరిమాణం(int16_t)},
    {GGUF_TYPE_UINT32,  పరిమాణం(uint32_t)},
    {GGUF_TYPE_INT32,   పరిమాణం(int32_t)},
    {GGUF_TYPE_FLOAT32, పరిమాణం(దశాంశం)},
    {GGUF_TYPE_BOOL,    పరిమాణం(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  పరిమాణం(uint64_t)},
    {GGUF_TYPE_INT64,   పరిమాణం(int64_t)},
    {GGUF_TYPE_FLOAT64, పరిమాణం(ద్విగుణ)},
};
static_assert(GGUF_TYPE_లెక్క == 13, "GGUF_TYPE_COUNT != 13");

స్థిర స్థిరరాశి std::map<gguf_type, స్థిరరాశి అక్షరం *> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8,   "u8"},
    {GGUF_TYPE_INT8,    "i8"},
    {GGUF_TYPE_UINT16,  "u16"},
    {GGUF_TYPE_INT16,   "i16"},
    {GGUF_TYPE_UINT32,  "u32"},
    {GGUF_TYPE_INT32,   "i32"},
    {GGUF_TYPE_FLOAT32, "f32"},
    {GGUF_TYPE_BOOL,    "bool"},
    {GGUF_TYPE_STRING,  "str"},
    {GGUF_TYPE_ARRAY,   "arr"},
    {GGUF_TYPE_UINT64,  "u64"},
    {GGUF_TYPE_INT64,   "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(GGUF_TYPE_లెక్క == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_కొలత(గణన gguf_type type) {
    స్వయంచాలక it = GGUF_TYPE_కొలత.find(type);
    తిరిగి it == GGUF_TYPE_కొలత.end() ? 0 : it->second;
}

నిర్మాణం gguf_kv {
    std::string కీలకం;

    తార్కిక is_array;
    గణన gguf_type type;

    std::vector<int8_t>      దత్తాంశం;
    std::vector<std::string> దత్తాంశం_string;

    మూస <రకం_పేరు T>
    gguf_kv(స్థిరరాశి std::string & కీలకం, స్థిరరాశి T value)
            : కీలకం(కీలకం), is_array(అబద్ధం), type(type_గమ్యం_gguf_type<T>::value) {
        GGML_ASSERT(!కీలకం.ఖాళీలు());
        దత్తాంశం.resize(పరిమాణం(T));
        జ్ఞాపకం_నకలు(దత్తాంశం.దత్తాంశం(), &value, పరిమాణం(T));
    }

    మూస <రకం_పేరు T>
    gguf_kv(స్థిరరాశి std::string & కీలకం, స్థిరరాశి std::vector<T> & value)
            : కీలకం(కీలకం), is_array(నిజం), type(type_గమ్యం_gguf_type<T>::value) {
        GGML_ASSERT(!కీలకం.ఖాళీలు());
        దత్తాంశం.resize(value.size()*పరిమాణం(T));
        కొరకు (size_t i = 0; i < value.size(); ++i) {
            స్థిరరాశి T tmp = value[i];
            జ్ఞాపకం_నకలు(దత్తాంశం.దత్తాంశం() + i*పరిమాణం(T), &tmp, పరిమాణం(T));
        }
    }

    gguf_kv(స్థిరరాశి std::string & కీలకం, స్థిరరాశి std::string & value)
            : కీలకం(కీలకం), is_array(అబద్ధం), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!కీలకం.ఖాళీలు());
        దత్తాంశం_string.push_back(value);
    }

    gguf_kv(స్థిరరాశి std::string & కీలకం, స్థిరరాశి std::vector<std::string> & value)
            : కీలకం(కీలకం), is_array(నిజం), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!కీలకం.ఖాళీలు());
        దత్తాంశం_string = value;
    }

    స్థిరరాశి std::string & get_కీలకం() స్థిరరాశి {
        తిరిగి కీలకం;
    }

    స్థిరరాశి గణన gguf_type & get_type() స్థిరరాశి {
        తిరిగి type;
    }

    size_t get_ne() స్థిరరాశి {
        ఒకవేళ (type == GGUF_TYPE_STRING) {
            స్థిరరాశి size_t ne = దత్తాంశం_string.size();
            GGML_ASSERT(is_array || ne == 1);
            తిరిగి ne;
        }
        స్థిరరాశి size_t type_కొలత = gguf_type_కొలత(type);
        GGML_ASSERT(దత్తాంశం.size() % type_కొలత == 0);
        స్థిరరాశి size_t ne = దత్తాంశం.size() / type_కొలత;
        GGML_ASSERT(is_array || ne == 1);
        తిరిగి ne;
    }

    మూస <రకం_పేరు T>
    స్థిరరాశి T & get_val(స్థిరరాశి size_t i = 0) స్థిరరాశి {
        GGML_ASSERT(type_గమ్యం_gguf_type<T>::value == type);
        ఒకవేళ constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(దత్తాంశం_string.size() >= i+1);
            తిరిగి దత్తాంశం_string[i];
        }
        స్థిరరాశి size_t type_కొలత = gguf_type_కొలత(type);
        GGML_ASSERT(దత్తాంశం.size() % type_కొలత == 0);
        GGML_ASSERT(దత్తాంశం.size() >= (i+1)*type_కొలత);
        తిరిగి reinterpret_cast<స్థిరరాశి T *>(దత్తాంశం.దత్తాంశం())[i];
    }

    శూన్య cast(స్థిరరాశి గణన gguf_type new_type) {
        స్థిరరాశి size_t new_type_కొలత = gguf_type_కొలత(new_type);
        GGML_ASSERT(దత్తాంశం.size() % new_type_కొలత == 0);
        type = new_type;
    }
};

నిర్మాణం gguf_tensor_info {
    నిర్మాణం ggml_tensor t; // for holding the equivalent info
    uint64_t స్థానభ్రంశం;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

నిర్మాణం gguf_context {
    uint32_t సంస్కరణ_2 = GGUF_సంస్కరణ;

    std::vector<నిర్మాణం gguf_kv> kv;
    std::vector<నిర్మాణం gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t స్థానభ్రంశం    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    శూన్య * దత్తాంశం = శూన్యసూచిక;
};

నిర్మాణం gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            శూన్య * userdata,
            size_t max_chunk_read,
            uint64_t దత్తాంశం_స్థానభ్రంశం = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          ప్రారంభం_స్థానభ్రంశం(దత్తాంశం_స్థానభ్రంశం),
          దత్తాంశం_స్థానభ్రంశం(దత్తాంశం_స్థానభ్రంశం),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    స్థిర uint64_t file_remain(FILE * file) {
        స్థిరరాశి int64_t cur = gguf_ftell(file);
        ఒకవేళ (cur < 0) {
            తిరిగి 0;
        }
        ఒకవేళ (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            తిరిగి 0;
        }
        స్థిరరాశి int64_t end = gguf_ftell(file);
        ఒకవేళ (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            తిరిగి 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        తిరిగి static_cast<uint64_t>(end - cur);
    }

    మూస <రకం_పేరు T>
    తార్కిక పఠ(T & dst) స్థిరరాశి {
        స్థిరరాశి size_t size = పరిమాణం(dst);
        ఒకవేళ (size > nbytes_remain) {
            తిరిగి అబద్ధం;
        }
        తిరిగి read_raw(&dst, size) == size;
    }

    మూస <రకం_పేరు T>
    తార్కిక పఠ(std::vector<T> & dst, స్థిరరాశి size_t n) స్థిరరాశి {
        ఒకవేళ (n > GGUF_MAX_ARRAY_ELEMENTS) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            ఒకవేళ (n > SIZE_MAX / పరిమాణం(uint64_t)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (nbytes_remain < n * పరిమాణం(uint64_t)) {
                తిరిగి అబద్ధం;
            }
        } లేకపోతే {
            ఒకవేళ (n > SIZE_MAX / పరిమాణం(T)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (nbytes_remain < n * పరిమాణం(T)) {
                తిరిగి అబద్ధం;
            }
        }
        dst.resize(n);
        కొరకు (size_t i = 0; i < dst.size(); ++i) {
            ఒకవేళ constexpr (std::is_same<T, తార్కిక>::value) {
                తార్కిక tmp;
                ఒకవేళ (!పఠ(tmp)) {
                    తిరిగి అబద్ధం;
                }
                dst[i] = tmp;
            } లేకపోతే {
                ఒకవేళ (!పఠ(dst[i])) {
                    తిరిగి అబద్ధం;
                }
            }
        }
        తిరిగి నిజం;
    }

    తార్కిక పఠ(తార్కిక & dst) స్థిరరాశి {
        int8_t tmp = -1;
        ఒకవేళ (!పఠ(tmp)) {
            తిరిగి అబద్ధం;
        }
        dst = tmp != 0;
        తిరిగి నిజం;
    }

    తార్కిక పఠ(గణన ggml_type & dst) స్థిరరాశి {
        int32_t tmp = -1;
        ఒకవేళ (!పఠ(tmp)) {
            తిరిగి అబద్ధం;
        }
        dst = ggml_type(tmp);
        తిరిగి నిజం;
    }

    తార్కిక పఠ(గణన gguf_type & dst) స్థిరరాశి {
        int32_t tmp = -1;
        ఒకవేళ (!పఠ(tmp)) {
            తిరిగి అబద్ధం;
        }
        dst = gguf_type(tmp);
        తిరిగి నిజం;
    }

    తార్కిక పఠ(std::string & dst) స్థిరరాశి {
        uint64_t size = 0;
        ఒకవేళ (!పఠ(size)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (size > GGUF_MAX_STRING_నిడివి) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_నిడివి);
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            తిరిగి అబద్ధం;
        }
        dst.resize(static_cast<size_t>(size));
        తిరిగి read_raw(dst.దత్తాంశం(), static_cast<size_t>(size)) == size;
    }

    తార్కిక పఠ(శూన్య * dst, స్థిరరాశి size_t size) స్థిరరాశి {
        ఒకవేళ (size > nbytes_remain) {
            తిరిగి అబద్ధం;
        }
        తిరిగి read_raw(dst, size) == size;
    }

    uint64_t tell() స్థిరరాశి {
        తిరిగి దత్తాంశం_స్థానభ్రంశం;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() స్థిరరాశి {
        తిరిగి ప్రారంభం_స్థానభ్రంశం;
    }

    తార్కిక seek(uint64_t absolute_స్థానభ్రంశం) స్థిరరాశి {
        స్థిరరాశి uint64_t చివర_స్థానభ్రంశం = uint64_t(దత్తాంశం_స్థానభ్రంశం) + nbytes_remain;
        ఒకవేళ (absolute_స్థానభ్రంశం > చివర_స్థానభ్రంశం) {
            తిరిగి అబద్ధం;
        }

        దత్తాంశం_స్థానభ్రంశం = absolute_స్థానభ్రంశం;
        nbytes_remain = చివర_స్థానభ్రంశం - absolute_స్థానభ్రంశం;

        తిరిగి నిజం;
    }

అంతరంగ:
    size_t read_raw(శూన్య * dst, size_t size) స్థిరరాశి {
        ఒకవేళ (callback == శూన్యసూచిక || size == 0) {
            తిరిగి 0;
        }

        uint8_t * దత్తాంశం = static_cast<uint8_t *>(dst);
        size_t మొత్తం_nread = 0;
        తార్కిక reached_eof = అబద్ధం;

        అయితే (మొత్తం_nread < size) {
            స్థిరరాశి size_t chunk_కొలత = std::min(max_chunk_read, size - మొత్తం_nread);
            ఒకవేళ (దత్తాంశం_స్థానభ్రంశం + మొత్తం_nread < దత్తాంశం_స్థానభ్రంశం) {
                ఆపు;
            }
            స్థిరరాశి size_t nread = callback(userdata, static_cast<శూన్య *>(దత్తాంశం + మొత్తం_nread), దత్తాంశం_స్థానభ్రంశం + మొత్తం_nread, chunk_కొలత);
            మొత్తం_nread += nread;
            ఒకవేళ (nread != chunk_కొలత) {
                reached_eof = నిజం;
                ఆపు;
            }
        }

        దత్తాంశం_స్థానభ్రంశం += మొత్తం_nread;
        GGML_ASSERT(మొత్తం_nread <= nbytes_remain);
        nbytes_remain -= మొత్తం_nread;

        ఒకవేళ (reached_eof) {
            nbytes_remain = 0;
        }

        తిరిగి మొత్తం_nread;
    }

    gguf_reader_callback_t callback = శూన్యసూచిక;
    శూన్య * userdata = శూన్యసూచిక;
    size_t max_chunk_read = 0;
    uint64_t ప్రారంభం_స్థానభ్రంశం = 0;
    మార్చదగిన uint64_t దత్తాంశం_స్థానభ్రంశం = 0;
    మార్చదగిన uint64_t nbytes_remain = 0;
};

నిర్మాణం gguf_context * gguf_init_ఖాళీలు(శూన్య) {
    తిరిగి కొత్త gguf_context;
}

మూస<రకం_పేరు T>
తార్కిక gguf_read_emplace_helper(స్థిరరాశి నిర్మాణం gguf_reader & gr, std::vector<నిర్మాణం gguf_kv> & kv, స్థిరరాశి std::string & కీలకం, స్థిరరాశి తార్కిక is_array, స్థిరరాశి size_t n) {
    ఒకవేళ (is_array) {
        std::vector<T> value;
        ప్రయత్నించు {
            ఒకవేళ (!gr.పఠ(value, n)) {
                తిరిగి అబద్ధం;
            }
        } పట్టుకో (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, కీలకం.c_str());
            తిరిగి అబద్ధం;
        } పట్టుకో (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, కీలకం.c_str());
            తిరిగి అబద్ధం;
        }
        kv.emplace_back(కీలకం, value);
    } లేకపోతే {
        T value;
        ఒకవేళ (!gr.పఠ(value)) {
            తిరిగి అబద్ధం;
        }
        kv.emplace_back(కీలకం, value);
    }
    తిరిగి నిజం;
}

స్థిర నిర్మాణం gguf_context * gguf_init_from_reader(స్థిరరాశి నిర్మాణం gguf_reader & gr, నిర్మాణం gguf_init_params params) {
    నిర్మాణం gguf_context * ctx = కొత్త gguf_context;

    తార్కిక ok = నిజం;

    // file magic
    {
        std::vector<అక్షరం> magic;
        ok = ok && gr.పఠ(magic, 4);

        ఒకవేళ (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            తిరిగి శూన్యసూచిక;
        }

        కొరకు (uint32_t i = 0; i < magic.size(); i++) {
            ఒకవేళ (magic[i] != GGUF_MAGIC[i]) {
                అక్షరం c0 = isprint(magic[0]) ? magic[0] : '?';
                అక్షరం c1 = isprint(magic[1]) ? magic[1] : '?';
                అక్షరం c2 = isprint(magic[2]) ? magic[2] : '?';
                అక్షరం c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }
        }
    }

    //శీర్షిక
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    ఒకవేళ (ok && gr.పఠ(ctx->సంస్కరణ_2)) {
        ఒకవేళ (ok && ctx->సంస్కరణ_2 == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->సంస్కరణ_2);
            ok = అబద్ధం;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        ఒకవేళ (ok && (ctx->సంస్కరణ_2 & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->సంస్కరణ_2);
            ok = అబద్ధం;
        }

        ఒకవేళ (ok && ctx->సంస్కరణ_2 == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = అబద్ధం;
        }
        ఒకవేళ (ok && ctx->సంస్కరణ_2 > GGUF_సంస్కరణ) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->సంస్కరణ_2, GGUF_సంస్కరణ);
            ok = అబద్ధం;
        }
    } లేకపోతే {
        ok = అబద్ధం;
    }

    ఒకవేళ (ok && gr.పఠ(n_tensors)) {
        static_assert(పరిమాణం(size_t) <= 8 && పరిమాణం(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ఒకవేళ (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/పరిమాణం(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/పరిమాణం(gguf_tensor_info));
            ok = అబద్ధం;
        }
    } లేకపోతే {
        ok = అబద్ధం;
    }

    ఒకవేళ (ok && gr.పఠ(n_kv)) {
        static_assert(పరిమాణం(size_t) <= 8 && పరిమాణం(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ఒకవేళ (n_kv < 0 || n_kv > int64_t(SIZE_MAX/పరిమాణం(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/పరిమాణం(gguf_kv));
            ok = అబద్ధం;
        }
    } లేకపోతే {
        ok = అబద్ధం;
    }

    ఒకవేళ (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        తిరిగి శూన్యసూచిక;
    }

    // KV pairs
    {
        std::unordered_set<std::string> చూసిన_కీలకాలు;

        కొరకు (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string కీలకం;
            gguf_type   type     = gguf_type(-1);
            తార్కిక        is_array = అబద్ధం;
            uint64_t    n        = 1;

            ప్రయత్నించు {
                ok = ok && gr.పఠ(కీలకం);
            } పట్టుకో (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = అబద్ధం;
            } పట్టుకో (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = అబద్ధం;
            }
            ఒకవేళ (ok && కీలకం.ఖాళీలు()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = అబద్ధం;
            }
            ఒకవేళ (ok && !చూసిన_కీలకాలు.insert(కీలకం).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, కీలకం.c_str(), i);
                ok = అబద్ధం;
            }
            ఒకవేళ (!ok) {
                ఆపు;
            }

            ok = ok && gr.పఠ(type);
            ఒకవేళ (type == GGUF_TYPE_ARRAY) {
                is_array = నిజం;
                ok = ok && gr.పఠ(type);
                ok = ok && gr.పఠ(n);
            }
            ఒకవేళ (!ok) {
                ఆపు;
            }

            ఎంపిక (type) {
                స్థితి GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<దశాంశం>      (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<తార్కిక>       (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<ద్విగుణ>     (gr, ctx->kv, కీలకం, is_array, n); ఆపు;
                స్థితి GGUF_TYPE_ARRAY:
                అప్రమేయ:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, కీలకం.c_str(), type);
                        ok = అబద్ధం;
                    } ఆపు;
            }
        }

        ఒకవేళ (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            తిరిగి శూన్యసూచిక;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        స్థిరరాశి పూర్ణాంకం alignment_idx = gguf_కనుగొను_కీలకం(ctx, GGUF_కీలకం_GENERAL_ALIGNMENT);
        ఒకవేళ (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_కీలకం_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            తిరిగి శూన్యసూచిక;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        ఒకవేళ (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            తిరిగి శూన్యసూచిక;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> చూసిన_tensor_పేర్లు;

    కొరకు (int64_t i = 0; ok && i < n_tensors; ++i) {
        నిర్మాణం gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            ప్రయత్నించు {
                ok = ok && gr.పఠ(name);
            } పట్టుకో (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = అబద్ధం;
            } పట్టుకో (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = అబద్ధం;
            }
            ఒకవేళ (name.నిడివి() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.నిడివి(), GGML_MAX_NAME);
                ok = అబద్ధం;
                ఆపు;
            }
            ggml_సమితి_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            ఒకవేళ (ok && !చూసిన_tensor_పేర్లు.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (!ok) {
            ఆపు;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.పఠ(n_dims);
            ఒకవేళ (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = అబద్ధం;
                ఆపు;
            }
            కొరకు (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                ఒకవేళ (j < n_dims) {
                    ok = ok && gr.పఠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                ఒకవేళ (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = అబద్ధం;
                    ఆపు;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            ఒకవేళ (ok) {
                int64_t current_nelements = info.t.ne[0];
                కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; ++i) {
                    ఒకవేళ (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        ఒకవేళ (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = అబద్ధం;
                             ఆపు;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 ఒకవేళ (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     ఆపు;
                 }
            }
        }
        ఒకవేళ (!ok) {
            ఆపు;
        }

        // tensor type
        {
            ok = ok && gr.పఠ(info.t.type);

            // check that tensor type is within defined range
            ఒకవేళ (info.t.type < 0 || info.t.type >= GGML_TYPE_లెక్క) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_లెక్క);
                ok = అబద్ధం;
                ఆపు;
            }
            స్థిరరాశి size_t  type_కొలత = ggml_type_కొలత(info.t.type);
            స్థిరరాశి int64_t blck_కొలత = ggml_blck_కొలత(info.t.type);

            // check that row size is divisible by block size
            ఒకవేళ (blck_కొలత == 0 || info.t.ne[0] % blck_కొలత != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (పూర్ణాంకం) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_కొలత);
                ok = అబద్ధం;
                ఆపు;
            }

            // check that the size of the tensor in bytes is representable
            ఒకవేళ (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_కొలత(info.t.type)) > SIZE_MAX/ggml_type_కొలత(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = అబద్ధం;
                ఆపు;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_కొలత;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_కొలత);
            కొరకు (పూర్ణాంకం j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        ఒకవేళ (!ok) {
            ఆపు;
        }

        // tensor data offset within buffer
        ok = ok && gr.పఠ(info.స్థానభ్రంశం);

        ctx->info.push_back(info);
    }

    ఒకవేళ (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        తిరిగి శూన్యసూచిక;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    ఒకవేళ (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        తిరిగి శూన్యసూచిక;
    }

    // store the current file offset - this is where the data section starts
    ctx->స్థానభ్రంశం = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        కొరకు (size_t i = 0; i < ctx->info.size(); ++i) {
            స్థిరరాశి gguf_tensor_info & ti = ctx->info[i];
            ఒకవేళ (ti.స్థానభ్రంశం != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.స్థానభ్రంశం, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }
            స్థిరరాశి size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            ఒకవేళ (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }
            స్థిరరాశి size_t padded_కొలత = GGML_PAD(nbytes, ctx->alignment);
            ఒకవేళ (SIZE_MAX - ctx->size < padded_కొలత) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_కొలత);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }
            ctx->size += padded_కొలత;
        }
    }

    // load the tensor data only if requested
    ఒకవేళ (params.ctx != శూన్యసూచిక) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_కొలత = 0;
        ఒకవేళ (params.no_alloc) {
            ఒకవేళ (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }

            స్థిరరాశి size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_కొలత = overhead;
        } లేకపోతే {
            ఒకవేళ ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }

            స్థిరరాశి size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            ఒకవేళ (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }

            mem_కొలత = overhead + ctx->size;
        }

        నిర్మాణం ggml_init_params pdata = {
            /*mem_size   =*/ mem_కొలత,
            /*mem_buffer =*/ శూన్యసూచిక,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        ఒకవేళ (*params.ctx == శూన్యసూచిక) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            తిరిగి శూన్యసూచిక;
        }

        నిర్మాణం ggml_context * ctx_దత్తాంశం = *params.ctx;

        నిర్మాణం ggml_tensor * దత్తాంశం = శూన్యసూచిక;

        ఒకవేళ (!params.no_alloc) {
            దత్తాంశం = ggml_new_tensor_1d(ctx_దత్తాంశం, GGML_TYPE_I8, ctx->size);

            ok = ok && దత్తాంశం != శూన్యసూచిక;

            ఒకవేళ (ok) {
                ggml_సమితి_name(దత్తాంశం, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.పఠ(దత్తాంశం->దత్తాంశం, ctx->size);

            ఒకవేళ (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_దత్తాంశం);
                *params.ctx = శూన్యసూచిక;
                gguf_free(ctx);
                తిరిగి శూన్యసూచిక;
            }

            ctx->దత్తాంశం = దత్తాంశం->దత్తాంశం;
        }

        ggml_సమితి_no_alloc(ctx_దత్తాంశం, నిజం);

        // create the tensors
        కొరకు (size_t i = 0; i < ctx->info.size(); ++i) {
            స్థిరరాశి నిర్మాణం gguf_tensor_info & info = ctx->info[i];

            నిర్మాణం ggml_tensor * cur = ggml_new_tensor(ctx_దత్తాంశం, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != శూన్యసూచిక;

            ఒకవేళ (!ok) {
                ఆపు;
            }

            ggml_సమితి_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            ఒకవేళ (!params.no_alloc) {
                cur->దత్తాంశం = (అక్షరం *) దత్తాంశం->దత్తాంశం + info.స్థానభ్రంశం;
            }
        }

        ఒకవేళ (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_దత్తాంశం);
            *params.ctx = శూన్యసూచిక;
            gguf_free(ctx);
            తిరిగి శూన్యసూచిక;
        }

        ggml_సమితి_no_alloc(ctx_దత్తాంశం, params.no_alloc);
    }

    తిరిగి ctx;
}

నిర్మాణం gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, శూన్య * userdata, size_t max_chunk_read, uint64_t max_expected_కొలత, నిర్మాణం gguf_init_params params) {
    ఒకవేళ (callback == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }

    స్థిరరాశి నిర్మాణం gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_కొలత);
    తిరిగి gguf_init_from_reader(gr, params);
}

నిర్మాణం gguf_file_reader {
    FILE * file;
    uint64_t స్థానభ్రంశం;
};

స్థిర size_t gguf_file_reader_callback(శూన్య * userdata, శూన్య * output, uint64_t స్థానభ్రంశం, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    ఒకవేళ (reader.స్థానభ్రంశం != స్థానభ్రంశం) {
        ఒకవేళ (స్థానభ్రంశం > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(స్థానభ్రంశం), SEEK_SET) != 0) {
            తిరిగి 0;
        }

        reader.స్థానభ్రంశం = స్థానభ్రంశం;
    }

    స్థిరరాశి size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.స్థానభ్రంశం += nread;
    తిరిగి nread;
}

నిర్మాణం gguf_context * gguf_init_from_file_ptr(FILE * file, నిర్మాణం gguf_init_params params) {
    ఒకవేళ (!file) {
        తిరిగి శూన్యసూచిక;
    }

    స్థిరరాశి int64_t cur = gguf_ftell(file);
    ఒకవేళ (cur < 0) {
        తిరిగి శూన్యసూచిక;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    స్థిరరాశి నిర్మాణం gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.స్థానభ్రంశం, gguf_reader::file_remain(file));
    తిరిగి gguf_init_from_reader(gr, params);
}

నిర్మాణం gguf_బఫరు_reader {
    స్థిరరాశి uint8_t * దత్తాంశం;
    size_t          size;
};

స్థిర size_t gguf_బఫరు_reader_callback(శూన్య * userdata, శూన్య * output, uint64_t స్థానభ్రంశం, size_t len) {
    GGML_ASSERT(len > 0);

    స్థిరరాశి gguf_బఫరు_reader & reader = *static_cast<gguf_బఫరు_reader *>(userdata);

    ఒకవేళ (స్థానభ్రంశం > reader.size || len > reader.size - స్థానభ్రంశం) {
        తిరిగి 0;
    }

    స్థిరరాశి size_t దత్తాంశం_స్థానభ్రంశం = static_cast<size_t>(స్థానభ్రంశం);
    స్థిరరాశి size_t nread = std::min(len, reader.size - దత్తాంశం_స్థానభ్రంశం);
    జ్ఞాపకం_నకలు(static_cast<uint8_t *>(output), reader.దత్తాంశం + దత్తాంశం_స్థానభ్రంశం, nread);
    తిరిగి nread;
}

నిర్మాణం gguf_context * gguf_init_from_బఫరు(స్థిరరాశి శూన్య * దత్తాంశం, size_t size, నిర్మాణం gguf_init_params params) {
    ఒకవేళ (దత్తాంశం == శూన్యసూచిక || size == 0) {
        తిరిగి శూన్యసూచిక;
    }

    gguf_బఫరు_reader reader = {
        /*.data = */ static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం),
        /*.size = */ size,
    };
    స్థిరరాశి నిర్మాణం gguf_reader gr(gguf_బఫరు_reader_callback, &reader, SIZE_MAX, 0, size);
    తిరిగి gguf_init_from_reader(gr, params);
}

నిర్మాణం gguf_context * gguf_init_from_file(స్థిరరాశి అక్షరం * fname, నిర్మాణం gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    ఒకవేళ (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        తిరిగి శూన్యసూచిక;
    }

    నిర్మాణం gguf_context * ఫలితం = gguf_init_from_file_ptr(file, params);
    fclose(file);
    తిరిగి ఫలితం;
}

శూన్య gguf_free(నిర్మాణం gguf_context * ctx) {
    ఒకవేళ (ctx == శూన్యసూచిక) {
        తిరిగి;
    }
    తొలగించు ctx;
}

స్థిరరాశి అక్షరం * gguf_type_name(గణన gguf_type type) {
    స్వయంచాలక it = GGUF_TYPE_NAME.find(type);
    తిరిగి it == GGUF_TYPE_NAME.end() ? శూన్యసూచిక : it->second;
}

uint32_t gguf_get_సంస్కరణ(స్థిరరాశి నిర్మాణం gguf_context * ctx) {
    తిరిగి ctx->సంస్కరణ_2;
}

size_t gguf_get_alignment(స్థిరరాశి నిర్మాణం gguf_context * ctx) {
    తిరిగి ctx->alignment;
}

size_t gguf_get_దత్తాంశం_స్థానభ్రంశం(స్థిరరాశి నిర్మాణం gguf_context * ctx) {
    తిరిగి ctx->స్థానభ్రంశం;
}

int64_t gguf_get_n_kv(స్థిరరాశి నిర్మాణం gguf_context * ctx) {
    తిరిగి ctx->kv.size();
}

int64_t gguf_కనుగొను_కీలకం(స్థిరరాశి నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం) {
    // return -1 if key not found
    int64_t keyfound = -1;

    స్థిరరాశి int64_t n_kv = gguf_get_n_kv(ctx);

    కొరకు (int64_t i = 0; i < n_kv; ++i) {
        ఒకవేళ (strcmp(కీలకం, gguf_get_కీలకం(ctx, i)) == 0) {
            keyfound = i;
            ఆపు;
        }
    }

    తిరిగి keyfound;
}

స్థిరరాశి అక్షరం * gguf_get_కీలకం(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    తిరిగి ctx->kv[కీలకం_id].get_కీలకం().c_str();
}

గణన gguf_type gguf_get_kv_type(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    తిరిగి ctx->kv[కీలకం_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[కీలకం_id].get_type();
}

గణన gguf_type gguf_get_arr_type(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].is_array);
    తిరిగి ctx->kv[కీలకం_id].get_type();
}

స్థిరరాశి శూన్య * gguf_get_arr_దత్తాంశం(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_type() != GGUF_TYPE_STRING);
    తిరిగి ctx->kv[కీలకం_id].దత్తాంశం.దత్తాంశం();
}

స్థిరరాశి అక్షరం * gguf_get_arr_str(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id, size_t i) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_type() == GGUF_TYPE_STRING);
    తిరిగి ctx->kv[కీలకం_id].దత్తాంశం_string[i].c_str();
}

size_t gguf_get_arr_n(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));

    ఒకవేళ (ctx->kv[కీలకం_id].type == GGUF_TYPE_STRING) {
        తిరిగి ctx->kv[కీలకం_id].దత్తాంశం_string.size();
    }

    స్థిరరాశి size_t type_కొలత = gguf_type_కొలత(ctx->kv[కీలకం_id].type);
    GGML_ASSERT(ctx->kv[కీలకం_id].దత్తాంశం.size() % type_కొలత == 0);
    తిరిగి ctx->kv[కీలకం_id].దత్తాంశం.size() / type_కొలత;
}

uint8_t gguf_get_val_u8(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<int32_t>();
}

దశాంశం gguf_get_val_f32(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<దశాంశం>();
}

uint64_t gguf_get_val_u64(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<int64_t>();
}

ద్విగుణ gguf_get_val_f64(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<ద్విగుణ>();
}

తార్కిక gguf_get_val_bool(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<తార్కిక>();
}

స్థిరరాశి అక్షరం * gguf_get_val_str(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    తిరిగి ctx->kv[కీలకం_id].get_val<std::string>().c_str();
}

స్థిరరాశి శూన్య * gguf_get_val_దత్తాంశం(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t కీలకం_id) {
    GGML_ASSERT(కీలకం_id >= 0 && కీలకం_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[కీలకం_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[కీలకం_id].get_type() != GGUF_TYPE_STRING);
    తిరిగి ctx->kv[కీలకం_id].దత్తాంశం.దత్తాంశం();
}

int64_t gguf_get_n_tensors(స్థిరరాశి నిర్మాణం gguf_context * ctx) {
    తిరిగి ctx->info.size();
}

int64_t gguf_కనుగొను_tensor(స్థిరరాశి నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    స్థిరరాశి int64_t n_tensors = gguf_get_n_tensors(ctx);

    కొరకు (int64_t i = 0; i < n_tensors; ++i) {
        ఒకవేళ (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            ఆపు;
        }
    }

    తిరిగి tensor_id;
}

size_t gguf_get_tensor_స్థానభ్రంశం(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    తిరిగి ctx->info[tensor_id].స్థానభ్రంశం;
}

స్థిరరాశి అక్షరం * gguf_get_tensor_name(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    తిరిగి ctx->info[tensor_id].t.name;
}

స్థిరరాశి int64_t * gguf_get_tensor_ne(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    తిరిగి ctx->info[tensor_id].t.ne;
}

గణన ggml_type gguf_get_tensor_type(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    తిరిగి ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_కొలత(స్థిరరాశి నిర్మాణం gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    తిరిగి ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_కీలకం(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం) {
    స్థిరరాశి int64_t కీలకం_id = gguf_కనుగొను_కీలకం(ctx, కీలకం);
    ఒకవేళ (కీలకం_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + కీలకం_id);
    }
    తిరిగి కీలకం_id;
}

మూస<రకం_పేరు T>
స్థిర శూన్య gguf_check_reserved_కీలకాలు(స్థిరరాశి std::string & కీలకం, స్థిరరాశి T val) {
    ఒకవేళ (కీలకం == GGUF_కీలకం_GENERAL_ALIGNMENT) {
        ఒకవేళ constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_కీలకం_GENERAL_ALIGNMENT " must be power of 2");
        } లేకపోతే {
            GGML_వాడని(val);
            GGML_ABORT(GGUF_కీలకం_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

శూన్య gguf_సమితి_val_u8(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, uint8_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_i8(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, int8_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_u16(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, uint16_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_i16(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, int16_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_u32(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, uint32_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_i32(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, int32_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_f32(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, దశాంశం val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_u64(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, uint64_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_i64(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, int64_t val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_f64(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, ద్విగుణ val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_bool(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, తార్కిక val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, val);
}

శూన్య gguf_సమితి_val_str(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, స్థిరరాశి అక్షరం * val) {
    gguf_check_reserved_కీలకాలు(కీలకం, val);
    gguf_remove_కీలకం(ctx, కీలకం);
    ctx->kv.emplace_back(కీలకం, std::string(val));
}

శూన్య gguf_సమితి_arr_దత్తాంశం(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, గణన gguf_type type, స్థిరరాశి శూన్య * దత్తాంశం, size_t n) {
    gguf_check_reserved_కీలకాలు(కీలకం, దత్తాంశం);
    gguf_remove_కీలకం(ctx, కీలకం);

    స్థిరరాశి size_t nbytes = n*gguf_type_కొలత(type);
    std::vector<int8_t> tmp(nbytes);
    ఒకవేళ (!tmp.ఖాళీలు()) {
        జ్ఞాపకం_నకలు(tmp.దత్తాంశం(), దత్తాంశం, nbytes);
    }
    ctx->kv.emplace_back(కీలకం, tmp);
    ctx->kv.back().cast(type);
}

శూన్య gguf_సమితి_arr_str(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * కీలకం, స్థిరరాశి అక్షరం ** దత్తాంశం, size_t n) {
    gguf_check_reserved_కీలకాలు(కీలకం, దత్తాంశం);
    gguf_remove_కీలకం(ctx, కీలకం);

    std::vector<std::string> tmp(n);
    కొరకు (size_t i = 0; i < n; ++i) {
        tmp[i] = దత్తాంశం[i];
    }
    ctx->kv.emplace_back(కీలకం, tmp);
}

// set or add KV pairs from another context
శూన్య gguf_సమితి_kv(నిర్మాణం gguf_context * ctx, స్థిరరాశి నిర్మాణం gguf_context * src) {
    స్థిరరాశి int64_t n_kv = gguf_get_n_kv(src);
    కొరకు (int64_t i = 0; i < n_kv; ++i) {
        స్థిరరాశి నిర్మాణం gguf_kv & kv = src->kv[i];

        ఒకవేళ (!kv.is_array) {
            ఎంపిక (kv.get_type()) {
                స్థితి GGUF_TYPE_UINT8:   gguf_సమితి_val_u8  (ctx, kv.get_కీలకం().c_str(), kv.get_val<uint8_t>());             ఆపు;
                స్థితి GGUF_TYPE_INT8:    gguf_సమితి_val_i8  (ctx, kv.get_కీలకం().c_str(), kv.get_val<int8_t>());              ఆపు;
                స్థితి GGUF_TYPE_UINT16:  gguf_సమితి_val_u16 (ctx, kv.get_కీలకం().c_str(), kv.get_val<uint16_t>());            ఆపు;
                స్థితి GGUF_TYPE_INT16:   gguf_సమితి_val_i16 (ctx, kv.get_కీలకం().c_str(), kv.get_val<int16_t>());             ఆపు;
                స్థితి GGUF_TYPE_UINT32:  gguf_సమితి_val_u32 (ctx, kv.get_కీలకం().c_str(), kv.get_val<uint32_t>());            ఆపు;
                స్థితి GGUF_TYPE_INT32:   gguf_సమితి_val_i32 (ctx, kv.get_కీలకం().c_str(), kv.get_val<int32_t>());             ఆపు;
                స్థితి GGUF_TYPE_FLOAT32: gguf_సమితి_val_f32 (ctx, kv.get_కీలకం().c_str(), kv.get_val<దశాంశం>());               ఆపు;
                స్థితి GGUF_TYPE_UINT64:  gguf_సమితి_val_u64 (ctx, kv.get_కీలకం().c_str(), kv.get_val<uint64_t>());            ఆపు;
                స్థితి GGUF_TYPE_INT64:   gguf_సమితి_val_i64 (ctx, kv.get_కీలకం().c_str(), kv.get_val<int64_t>());             ఆపు;
                స్థితి GGUF_TYPE_FLOAT64: gguf_సమితి_val_f64 (ctx, kv.get_కీలకం().c_str(), kv.get_val<ద్విగుణ>());              ఆపు;
                స్థితి GGUF_TYPE_BOOL:    gguf_సమితి_val_bool(ctx, kv.get_కీలకం().c_str(), kv.get_val<తార్కిక>());                ఆపు;
                స్థితి GGUF_TYPE_STRING:  gguf_సమితి_val_str (ctx, kv.get_కీలకం().c_str(), kv.get_val<std::string>().c_str()); ఆపు;
                స్థితి GGUF_TYPE_ARRAY:
                అప్రమేయ: GGML_ABORT("invalid type");
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t ne = kv.get_ne();

        ఎంపిక (kv.get_type()) {
            స్థితి GGUF_TYPE_UINT8:
            స్థితి GGUF_TYPE_INT8:
            స్థితి GGUF_TYPE_UINT16:
            స్థితి GGUF_TYPE_INT16:
            స్థితి GGUF_TYPE_UINT32:
            స్థితి GGUF_TYPE_INT32:
            స్థితి GGUF_TYPE_FLOAT32:
            స్థితి GGUF_TYPE_UINT64:
            స్థితి GGUF_TYPE_INT64:
            స్థితి GGUF_TYPE_FLOAT64:
            స్థితి GGUF_TYPE_BOOL: {
                gguf_సమితి_arr_దత్తాంశం(ctx, kv.get_కీలకం().c_str(), kv.get_type(), kv.దత్తాంశం.దత్తాంశం(), ne);
            } ఆపు;
            స్థితి GGUF_TYPE_STRING: {
                std::vector<స్థిరరాశి అక్షరం *> tmp(ne);
                కొరకు (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.దత్తాంశం_string[j].c_str();
                }
                gguf_సమితి_arr_str(ctx, kv.get_కీలకం().c_str(), tmp.దత్తాంశం(), ne);
            } ఆపు;
            స్థితి GGUF_TYPE_ARRAY:
            అప్రమేయ: GGML_ABORT("invalid type");
        }
    }
}

శూన్య gguf_కూడు_tensor(
             నిర్మాణం gguf_context * ctx,
        స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    ఒకవేళ (gguf_కనుగొను_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    నిర్మాణం gguf_tensor_info ti;
    ti.t = *tensor;
    ti.స్థానభ్రంశం = ctx->info.ఖాళీలు() ? 0 :
        ctx->info.back().స్థానభ్రంశం + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

శూన్య gguf_సమితి_tensor_type(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * name, గణన ggml_type type) {
    స్థిరరాశి int64_t tensor_id = gguf_కనుగొను_tensor(ctx, name);
    ఒకవేళ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    నిర్మాణం ggml_tensor * tensor = &ctx->info[tensor_id].t;
    స్థిరరాశి size_t  type_కొలత = ggml_type_కొలత(type);
    స్థిరరాశి int64_t blck_కొలత = ggml_blck_కొలత(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_కొలత == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_కొలత;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_కొలత);
    కొరకు (పూర్ణాంకం i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    స్థిరరాశి int64_t n_tensors = gguf_get_n_tensors(ctx);
    కొరకు (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].స్థానభ్రంశం = ctx->info[i - 1].స్థానభ్రంశం + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

శూన్య gguf_సమితి_tensor_దత్తాంశం(నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * name, స్థిరరాశి శూన్య * దత్తాంశం) {
    స్థిరరాశి int64_t tensor_id = gguf_కనుగొను_tensor(ctx, name);
    ఒకవేళ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.దత్తాంశం = (శూన్య *)(uintptr_t)దత్తాంశం; // double cast suppresses warning about casting away const
}

నిర్మాణం gguf_writer_పునాది {
    size_t written_బైట్లు {0u};

    మాయా ~gguf_writer_పునాది(శూన్య) = అప్రమేయ;

    // we bet on devirtualization
    మాయా శూన్య లిఖ(int8_t val) = 0;
    మాయా శూన్య లిఖ(స్థిరరాశి std::vector<int8_t> & val) = 0;
    మాయా శూన్య write_tensor_దత్తాంశం(స్థిరరాశి నిర్మాణం gguf_tensor_info & info, size_t స్థానభ్రంశం_దత్తాంశం, size_t alignment) = 0;

    మూస <రకం_పేరు T>
    శూన్య లిఖ(స్థిరరాశి T & val) {
        కొరకు (size_t i = 0; i < పరిమాణం(val); ++i) {
            లిఖ(reinterpret_cast<స్థిరరాశి int8_t *>(&val)[i]);
        }
    }

    శూన్య లిఖ(స్థిరరాశి తార్కిక & val) {
        స్థిరరాశి int8_t val8 = val ? 1 : 0;
        లిఖ(val8);
    }

    శూన్య లిఖ(స్థిరరాశి std::string & val) {
        {
            స్థిరరాశి uint64_t n = val.నిడివి();
            లిఖ(n);
        }
        కొరకు (size_t i = 0; i < val.నిడివి(); ++i) {
            లిఖ((val.దత్తాంశం())[i]);
        }
    }

    శూన్య లిఖ(స్థిరరాశి అక్షరం * val) {
        లిఖ(std::string(val));
    }

    శూన్య లిఖ(స్థిరరాశి గణన ggml_type & val) {
        లిఖ(int32_t(val));
    }

    శూన్య లిఖ(స్థిరరాశి గణన gguf_type & val) {
        లిఖ(int32_t(val));
    }

    శూన్య లిఖ(స్థిరరాశి నిర్మాణం gguf_kv & kv) {
        స్థిరరాశి uint64_t ne = kv.get_ne();

        లిఖ(kv.get_కీలకం());

        ఒకవేళ (kv.is_array) {
            లిఖ(GGUF_TYPE_ARRAY);
            లిఖ(kv.get_type());
            లిఖ(ne);
        } లేకపోతే {
            లిఖ(kv.get_type());
        }

        ఎంపిక (kv.get_type()) {
            స్థితి GGUF_TYPE_UINT8:
            స్థితి GGUF_TYPE_INT8:
            స్థితి GGUF_TYPE_UINT16:
            స్థితి GGUF_TYPE_INT16:
            స్థితి GGUF_TYPE_UINT32:
            స్థితి GGUF_TYPE_INT32:
            స్థితి GGUF_TYPE_FLOAT32:
            స్థితి GGUF_TYPE_UINT64:
            స్థితి GGUF_TYPE_INT64:
            స్థితి GGUF_TYPE_FLOAT64: {
                లిఖ(kv.దత్తాంశం);
            } ఆపు;
            స్థితి GGUF_TYPE_BOOL: {
                కొరకు (size_t i = 0; i < ne; ++i) {
                    లిఖ(kv.get_val<తార్కిక>(i));
                }
            } ఆపు;
            స్థితి GGUF_TYPE_STRING: {
                కొరకు (size_t i = 0; i < ne; ++i) {
                    లిఖ(kv.get_val<std::string>(i));
                }
            } ఆపు;
            స్థితి GGUF_TYPE_ARRAY:
            అప్రమేయ: GGML_ABORT("invalid type");
        }
    }

    శూన్య write_tensor_meta(స్థిరరాశి నిర్మాణం gguf_tensor_info & info) {
        లిఖ(info.t.name);

        స్థిరరాశి uint32_t n_dims = ggml_n_dims(&info.t);
        లిఖ(n_dims);

        కొరకు (uint32_t j = 0; j < n_dims; ++j) {
            లిఖ(info.t.ne[j]);
        }
        లిఖ(info.t.type);
        లిఖ(info.స్థానభ్రంశం);
    }

    శూన్య pad(స్థిరరాశి size_t alignment) {
        అయితే (written_బైట్లు % alignment != 0) {
            స్థిరరాశి int8_t zero = 0;
            లిఖ(zero);
        }
    }
};

// vector buffer based writer
నిర్మాణం gguf_writer_buf final : బహిరంగ gguf_writer_పునాది {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    ఉపయోగించి gguf_writer_పునాది::లిఖ;

    శూన్య లిఖ(స్థిరరాశి int8_t val) override {
        buf.push_back(val);
        written_బైట్లు++;
    }

    శూన్య లిఖ(స్థిరరాశి std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_బైట్లు += val.size();
    }

    శూన్య write_tensor_దత్తాంశం(స్థిరరాశి నిర్మాణం gguf_tensor_info & info, స్థిరరాశి size_t స్థానభ్రంశం_దత్తాంశం, స్థిరరాశి size_t alignment) override {
        GGML_ASSERT(buf.size() - స్థానభ్రంశం_దత్తాంశం == info.స్థానభ్రంశం);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        స్థిరరాశి size_t స్థానభ్రంశం = buf.size();
        స్థిరరాశి size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(స్థానభ్రంశం + nbytes);
        ఒకవేళ (info.t.బఫరు) {
            ggml_backend_tensor_get(&info.t, buf.దత్తాంశం() + స్థానభ్రంశం, 0, nbytes);
        } లేకపోతే {
            GGML_ASSERT(info.t.దత్తాంశం);
            జ్ఞాపకం_నకలు(buf.దత్తాంశం() + స్థానభ్రంశం, info.t.దత్తాంశం, nbytes);
        }
        written_బైట్లు += nbytes;

        pad(alignment);
    }
};

// file based writer
నిర్మాణం gguf_writer_file final : బహిరంగ gguf_writer_పునాది {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    ఉపయోగించి gguf_writer_పునాది::లిఖ;

    శూన్య లిఖ(స్థిరరాశి int8_t val) override {
        స్థిరరాశి స్వయంచాలక real_val = static_cast<uint8_t>(val);
        స్థిరరాశి స్వయంచాలక ret = fputc(real_val, file);
        written_బైట్లు++;
        ఒకవేళ (ret != real_val) {
            విసురు std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((పూర్ణాంకం)real_val) + "'");
        }
    }

    శూన్య లిఖ(స్థిరరాశి std::vector<int8_t> & val) override {
        స్థిరరాశి స్వయంచాలక ret = fwrite(val.దత్తాంశం(), 1, val.size(), file);
        written_బైట్లు += val.size();
        ఒకవేళ (ret != val.size()) {
            విసురు std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    శూన్య write_tensor_దత్తాంశం(స్థిరరాశి నిర్మాణం gguf_tensor_info & info, స్థిరరాశి size_t స్థానభ్రంశం_దత్తాంశం, స్థిరరాశి size_t alignment) override {
        GGML_ASSERT(written_బైట్లు - స్థానభ్రంశం_దత్తాంశం == info.స్థానభ్రంశం);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        స్థిరరాశి size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        ఒకవేళ (info.t.బఫరు) {
            ggml_backend_tensor_get(&info.t, buf.దత్తాంశం(), 0, nbytes);
        } లేకపోతే {
            GGML_ASSERT(info.t.దత్తాంశం);
            జ్ఞాపకం_నకలు(buf.దత్తాంశం(), info.t.దత్తాంశం, nbytes);
        }
        లిఖ(buf);

        pad(alignment);
    }
};

మూస <రకం_పేరు writer_t>
స్థిర శూన్య gguf_write_బహిర్గతం(స్థిరరాశి నిర్మాణం gguf_context * ctx, writer_t & gw, తార్కిక only_meta) {
    స్థిరరాశి int64_t n_kv      = gguf_get_n_kv(ctx);
    స్థిరరాశి int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.లిఖ(GGUF_MAGIC[0]);
    gw.లిఖ(GGUF_MAGIC[1]);
    gw.లిఖ(GGUF_MAGIC[2]);
    gw.లిఖ(GGUF_MAGIC[3]);
    gw.లిఖ(ctx->సంస్కరణ_2);
    gw.లిఖ(n_tensors);
    gw.లిఖ(n_kv);

    // write key-value pairs
    కొరకు (int64_t i = 0; i < n_kv; ++i) {
        gw.లిఖ(ctx->kv[i]);
    }

    // write tensor info
    కొరకు (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    ఒకవేళ (only_meta) {
        తిరిగి;
    }

    స్థిరరాశి size_t స్థానభ్రంశం_దత్తాంశం = gw.written_బైట్లు;

    // write tensor data
    కొరకు (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_దత్తాంశం(ctx->info[i], స్థానభ్రంశం_దత్తాంశం, ctx->alignment);
    }
}

శూన్య gguf_write_గమ్యం_buf(స్థిరరాశి నిర్మాణం gguf_context * ctx, std::vector<int8_t> & buf, తార్కిక only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_బహిర్గతం(ctx, gw, only_meta);
}

తార్కిక gguf_write_గమ్యం_file_ptr(స్థిరరాశి నిర్మాణం gguf_context * ctx, FILE * file, తార్కిక only_meta) {
    GGML_ASSERT(file);

    ప్రయత్నించు {
        gguf_writer_file gw(file);
        gguf_write_బహిర్గతం(ctx, gw, only_meta);
    } పట్టుకో (స్థిరరాశి std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

తార్కిక gguf_write_గమ్యం_file(స్థిరరాశి నిర్మాణం gguf_context * ctx, స్థిరరాశి అక్షరం * fname, తార్కిక only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    ఒకవేళ (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక success = gguf_write_గమ్యం_file_ptr(ctx, file, only_meta);
    ఒకవేళ (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    తిరిగి success;
}

size_t gguf_get_meta_కొలత(స్థిరరాశి నిర్మాణం gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_గమ్యం_buf(ctx, buf, /*only_meta =*/ నిజం);
    తిరిగి buf.size();
}

శూన్య gguf_get_meta_దత్తాంశం(స్థిరరాశి నిర్మాణం gguf_context * ctx, శూన్య * దత్తాంశం) {
    std::vector<int8_t> buf;
    gguf_write_గమ్యం_buf(ctx, buf, /*only_meta =*/ నిజం);
    జ్ఞాపకం_నకలు(దత్తాంశం, buf.దత్తాంశం(), buf.size());
}
