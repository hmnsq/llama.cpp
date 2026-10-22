#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "gguf.h"

#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <cerrno>
#অন্তর্ভাব <map>
#অন্তর্ভাব <new>
#অন্তর্ভাব <stdexcept>
#অন্তর্ভাব <string>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>

#define GGUF_MAX_STRING_দৈর্ঘ্য  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#অন্যথা
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
কাঠামো type_to_gguf_type;

template <>
কাঠামো type_to_gguf_type<uint8_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT8;
};

template <>
কাঠামো type_to_gguf_type<int8_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT8;
};

template <>
কাঠামো type_to_gguf_type<uint16_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT16;
};

template <>
কাঠামো type_to_gguf_type<int16_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT16;
};

template <>
কাঠামো type_to_gguf_type<uint32_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT32;
};

template <>
কাঠামো type_to_gguf_type<int32_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT32;
};

template <>
কাঠামো type_to_gguf_type<দশমিক> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
কাঠামো type_to_gguf_type<যৌক্তিক> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_BOOL;
};

template <>
কাঠামো type_to_gguf_type<std::string> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_STRING;
};

template <>
কাঠামো type_to_gguf_type<uint64_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT64;
};

template <>
কাঠামো type_to_gguf_type<int64_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT64;
};

template <>
কাঠামো type_to_gguf_type<দ্বিগুণ> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_FLOAT64;
};

স্থির ধ্রুবক std::map<gguf_type, size_t> GGUF_TYPE_আকার = {
    {GGUF_TYPE_UINT8,   আকার(uint8_t)},
    {GGUF_TYPE_INT8,    আকার(int8_t)},
    {GGUF_TYPE_UINT16,  আকার(uint16_t)},
    {GGUF_TYPE_INT16,   আকার(int16_t)},
    {GGUF_TYPE_UINT32,  আকার(uint32_t)},
    {GGUF_TYPE_INT32,   আকার(int32_t)},
    {GGUF_TYPE_FLOAT32, আকার(দশমিক)},
    {GGUF_TYPE_BOOL,    আকার(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  আকার(uint64_t)},
    {GGUF_TYPE_INT64,   আকার(int64_t)},
    {GGUF_TYPE_FLOAT64, আকার(দ্বিগুণ)},
};
static_assert(GGUF_TYPE_গুনতি == 13, "GGUF_TYPE_COUNT != 13");

স্থির ধ্রুবক std::map<gguf_type, ধ্রুবক অক্ষর *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_গুনতি == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_আকার(গণনা gguf_type type) {
    auto it = GGUF_TYPE_আকার.find(type);
    ফেরত it == GGUF_TYPE_আকার.end() ? 0 : it->second;
}

কাঠামো gguf_kv {
    std::string চাবি;

    যৌক্তিক is_array;
    গণনা gguf_type type;

    std::vector<int8_t>      তথ্য;
    std::vector<std::string> তথ্য_string;

    template <typename T>
    gguf_kv(ধ্রুবক std::string & চাবি, ধ্রুবক T value)
            : চাবি(চাবি), is_array(মিথ্যা), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!চাবি.empty());
        তথ্য.resize(আকার(T));
        memcpy(তথ্য.তথ্য(), &value, আকার(T));
    }

    template <typename T>
    gguf_kv(ধ্রুবক std::string & চাবি, ধ্রুবক std::vector<T> & value)
            : চাবি(চাবি), is_array(সত্য), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!চাবি.empty());
        তথ্য.resize(value.size()*আকার(T));
        জন্য (size_t i = 0; i < value.size(); ++i) {
            ধ্রুবক T tmp = value[i];
            memcpy(তথ্য.তথ্য() + i*আকার(T), &tmp, আকার(T));
        }
    }

    gguf_kv(ধ্রুবক std::string & চাবি, ধ্রুবক std::string & value)
            : চাবি(চাবি), is_array(মিথ্যা), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!চাবি.empty());
        তথ্য_string.push_back(value);
    }

    gguf_kv(ধ্রুবক std::string & চাবি, ধ্রুবক std::vector<std::string> & value)
            : চাবি(চাবি), is_array(সত্য), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!চাবি.empty());
        তথ্য_string = value;
    }

    ধ্রুবক std::string & get_চাবি() ধ্রুবক {
        ফেরত চাবি;
    }

    ধ্রুবক গণনা gguf_type & get_type() ধ্রুবক {
        ফেরত type;
    }

    size_t get_ne() ধ্রুবক {
        যদি (type == GGUF_TYPE_STRING) {
            ধ্রুবক size_t ne = তথ্য_string.size();
            GGML_ASSERT(is_array || ne == 1);
            ফেরত ne;
        }
        ধ্রুবক size_t type_আকার = gguf_type_আকার(type);
        GGML_ASSERT(তথ্য.size() % type_আকার == 0);
        ধ্রুবক size_t ne = তথ্য.size() / type_আকার;
        GGML_ASSERT(is_array || ne == 1);
        ফেরত ne;
    }

    template <typename T>
    ধ্রুবক T & get_val(ধ্রুবক size_t i = 0) ধ্রুবক {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        যদি constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(তথ্য_string.size() >= i+1);
            ফেরত তথ্য_string[i];
        }
        ধ্রুবক size_t type_আকার = gguf_type_আকার(type);
        GGML_ASSERT(তথ্য.size() % type_আকার == 0);
        GGML_ASSERT(তথ্য.size() >= (i+1)*type_আকার);
        ফেরত reinterpret_cast<ধ্রুবক T *>(তথ্য.তথ্য())[i];
    }

    শূন্য cast(ধ্রুবক গণনা gguf_type new_type) {
        ধ্রুবক size_t new_type_আকার = gguf_type_আকার(new_type);
        GGML_ASSERT(তথ্য.size() % new_type_আকার == 0);
        type = new_type;
    }
};

কাঠামো gguf_tensor_info {
    কাঠামো ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

কাঠামো gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<কাঠামো gguf_kv> kv;
    std::vector<কাঠামো gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    শূন্য * তথ্য = nullptr;
};

কাঠামো gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            শূন্য * userdata,
            size_t max_chunk_read,
            uint64_t তথ্য_offset = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          start_offset(তথ্য_offset),
          তথ্য_offset(তথ্য_offset),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    স্থির uint64_t file_remain(FILE * file) {
        ধ্রুবক int64_t cur = gguf_ftell(file);
        যদি (cur < 0) {
            ফেরত 0;
        }
        যদি (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ফেরত 0;
        }
        ধ্রুবক int64_t end = gguf_ftell(file);
        যদি (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ফেরত 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        ফেরত static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    যৌক্তিক পঠ(T & dst) ধ্রুবক {
        ধ্রুবক size_t size = আকার(dst);
        যদি (size > nbytes_remain) {
            ফেরত মিথ্যা;
        }
        ফেরত read_raw(&dst, size) == size;
    }

    template <typename T>
    যৌক্তিক পঠ(std::vector<T> & dst, ধ্রুবক size_t n) ধ্রুবক {
        যদি (n > GGUF_MAX_ARRAY_ELEMENTS) {
            ফেরত মিথ্যা;
        }
        যদি constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            যদি (n > SIZE_MAX / আকার(uint64_t)) {
                ফেরত মিথ্যা;
            }
            যদি (nbytes_remain < n * আকার(uint64_t)) {
                ফেরত মিথ্যা;
            }
        } অন্যথা {
            যদি (n > SIZE_MAX / আকার(T)) {
                ফেরত মিথ্যা;
            }
            যদি (nbytes_remain < n * আকার(T)) {
                ফেরত মিথ্যা;
            }
        }
        dst.resize(n);
        জন্য (size_t i = 0; i < dst.size(); ++i) {
            যদি constexpr (std::is_same<T, যৌক্তিক>::value) {
                যৌক্তিক tmp;
                যদি (!পঠ(tmp)) {
                    ফেরত মিথ্যা;
                }
                dst[i] = tmp;
            } অন্যথা {
                যদি (!পঠ(dst[i])) {
                    ফেরত মিথ্যা;
                }
            }
        }
        ফেরত সত্য;
    }

    যৌক্তিক পঠ(যৌক্তিক & dst) ধ্রুবক {
        int8_t tmp = -1;
        যদি (!পঠ(tmp)) {
            ফেরত মিথ্যা;
        }
        dst = tmp != 0;
        ফেরত সত্য;
    }

    যৌক্তিক পঠ(গণনা ggml_type & dst) ধ্রুবক {
        int32_t tmp = -1;
        যদি (!পঠ(tmp)) {
            ফেরত মিথ্যা;
        }
        dst = ggml_type(tmp);
        ফেরত সত্য;
    }

    যৌক্তিক পঠ(গণনা gguf_type & dst) ধ্রুবক {
        int32_t tmp = -1;
        যদি (!পঠ(tmp)) {
            ফেরত মিথ্যা;
        }
        dst = gguf_type(tmp);
        ফেরত সত্য;
    }

    যৌক্তিক পঠ(std::string & dst) ধ্রুবক {
        uint64_t size = 0;
        যদি (!পঠ(size)) {
            ফেরত মিথ্যা;
        }
        যদি (size > GGUF_MAX_STRING_দৈর্ঘ্য) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_দৈর্ঘ্য);
            ফেরত মিথ্যা;
        }
        যদি (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            ফেরত মিথ্যা;
        }
        dst.resize(static_cast<size_t>(size));
        ফেরত read_raw(dst.তথ্য(), static_cast<size_t>(size)) == size;
    }

    যৌক্তিক পঠ(শূন্য * dst, ধ্রুবক size_t size) ধ্রুবক {
        যদি (size > nbytes_remain) {
            ফেরত মিথ্যা;
        }
        ফেরত read_raw(dst, size) == size;
    }

    uint64_t tell() ধ্রুবক {
        ফেরত তথ্য_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ধ্রুবক {
        ফেরত start_offset;
    }

    যৌক্তিক seek(uint64_t absolute_offset) ধ্রুবক {
        ধ্রুবক uint64_t end_offset = uint64_t(তথ্য_offset) + nbytes_remain;
        যদি (absolute_offset > end_offset) {
            ফেরত মিথ্যা;
        }

        তথ্য_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        ফেরত সত্য;
    }

ব্যক্তিগত:
    size_t read_raw(শূন্য * dst, size_t size) ধ্রুবক {
        যদি (callback == nullptr || size == 0) {
            ফেরত 0;
        }

        uint8_t * তথ্য = static_cast<uint8_t *>(dst);
        size_t যোগ_nread = 0;
        যৌক্তিক reached_eof = মিথ্যা;

        যখন (যোগ_nread < size) {
            ধ্রুবক size_t chunk_আকার = std::min(max_chunk_read, size - যোগ_nread);
            যদি (তথ্য_offset + যোগ_nread < তথ্য_offset) {
                বিরতি;
            }
            ধ্রুবক size_t nread = callback(userdata, static_cast<শূন্য *>(তথ্য + যোগ_nread), তথ্য_offset + যোগ_nread, chunk_আকার);
            যোগ_nread += nread;
            যদি (nread != chunk_আকার) {
                reached_eof = সত্য;
                বিরতি;
            }
        }

        তথ্য_offset += যোগ_nread;
        GGML_ASSERT(যোগ_nread <= nbytes_remain);
        nbytes_remain -= যোগ_nread;

        যদি (reached_eof) {
            nbytes_remain = 0;
        }

        ফেরত যোগ_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    শূন্য * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t তথ্য_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

কাঠামো gguf_context * gguf_init_empty(শূন্য) {
    ফেরত নতুন gguf_context;
}

template<typename T>
যৌক্তিক gguf_read_emplace_helper(ধ্রুবক কাঠামো gguf_reader & gr, std::vector<কাঠামো gguf_kv> & kv, ধ্রুবক std::string & চাবি, ধ্রুবক যৌক্তিক is_array, ধ্রুবক size_t n) {
    যদি (is_array) {
        std::vector<T> value;
        চেষ্টা {
            যদি (!gr.পঠ(value, n)) {
                ফেরত মিথ্যা;
            }
        } ধরো (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, চাবি.c_str());
            ফেরত মিথ্যা;
        } ধরো (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, চাবি.c_str());
            ফেরত মিথ্যা;
        }
        kv.emplace_back(চাবি, value);
    } অন্যথা {
        T value;
        যদি (!gr.পঠ(value)) {
            ফেরত মিথ্যা;
        }
        kv.emplace_back(চাবি, value);
    }
    ফেরত সত্য;
}

স্থির কাঠামো gguf_context * gguf_init_from_reader(ধ্রুবক কাঠামো gguf_reader & gr, কাঠামো gguf_init_params params) {
    কাঠামো gguf_context * ctx = নতুন gguf_context;

    যৌক্তিক ok = সত্য;

    // file magic
    {
        std::vector<অক্ষর> magic;
        ok = ok && gr.পঠ(magic, 4);

        যদি (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            ফেরত nullptr;
        }

        জন্য (uint32_t i = 0; i < magic.size(); i++) {
            যদি (magic[i] != GGUF_MAGIC[i]) {
                অক্ষর c0 = isprint(magic[0]) ? magic[0] : '?';
                অক্ষর c1 = isprint(magic[1]) ? magic[1] : '?';
                অক্ষর c2 = isprint(magic[2]) ? magic[2] : '?';
                অক্ষর c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                ফেরত nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    যদি (ok && gr.পঠ(ctx->version)) {
        যদি (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = মিথ্যা;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        যদি (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = মিথ্যা;
        }

        যদি (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = মিথ্যা;
        }
        যদি (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = মিথ্যা;
        }
    } অন্যথা {
        ok = মিথ্যা;
    }

    যদি (ok && gr.পঠ(n_tensors)) {
        static_assert(আকার(size_t) <= 8 && আকার(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        যদি (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/আকার(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/আকার(gguf_tensor_info));
            ok = মিথ্যা;
        }
    } অন্যথা {
        ok = মিথ্যা;
    }

    যদি (ok && gr.পঠ(n_kv)) {
        static_assert(আকার(size_t) <= 8 && আকার(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        যদি (n_kv < 0 || n_kv > int64_t(SIZE_MAX/আকার(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/আকার(gguf_kv));
            ok = মিথ্যা;
        }
    } অন্যথা {
        ok = মিথ্যা;
    }

    যদি (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        ফেরত nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_চাবিগুলি;

        জন্য (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string চাবি;
            gguf_type   type     = gguf_type(-1);
            যৌক্তিক        is_array = মিথ্যা;
            uint64_t    n        = 1;

            চেষ্টা {
                ok = ok && gr.পঠ(চাবি);
            } ধরো (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = মিথ্যা;
            } ধরো (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = মিথ্যা;
            }
            যদি (ok && চাবি.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = মিথ্যা;
            }
            যদি (ok && !seen_চাবিগুলি.insert(চাবি).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, চাবি.c_str(), i);
                ok = মিথ্যা;
            }
            যদি (!ok) {
                বিরতি;
            }

            ok = ok && gr.পঠ(type);
            যদি (type == GGUF_TYPE_ARRAY) {
                is_array = সত্য;
                ok = ok && gr.পঠ(type);
                ok = ok && gr.পঠ(n);
            }
            যদি (!ok) {
                বিরতি;
            }

            নির্বাচন (type) {
                ক্ষেত্র GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<দশমিক>      (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<যৌক্তিক>       (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<দ্বিগুণ>     (gr, ctx->kv, চাবি, is_array, n); বিরতি;
                ক্ষেত্র GGUF_TYPE_ARRAY:
                পূর্বনির্ধারিত:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, চাবি.c_str(), type);
                        ok = মিথ্যা;
                    } বিরতি;
            }
        }

        যদি (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            ফেরত nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ধ্রুবক পূর্ণসংখ্যা alignment_idx = gguf_find_চাবি(ctx, GGUF_চাবি_GENERAL_ALIGNMENT);
        যদি (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_চাবি_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            ফেরত nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        যদি (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            ফেরত nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    জন্য (int64_t i = 0; ok && i < n_tensors; ++i) {
        কাঠামো gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            চেষ্টা {
                ok = ok && gr.পঠ(name);
            } ধরো (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = মিথ্যা;
            } ধরো (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = মিথ্যা;
            }
            যদি (name.দৈর্ঘ্য() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.দৈর্ঘ্য(), GGML_MAX_NAME);
                ok = মিথ্যা;
                বিরতি;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            যদি (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = মিথ্যা;
                বিরতি;
            }
        }
        যদি (!ok) {
            বিরতি;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.পঠ(n_dims);
            যদি (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = মিথ্যা;
                বিরতি;
            }
            জন্য (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                যদি (j < n_dims) {
                    ok = ok && gr.পঠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                যদি (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = মিথ্যা;
                    বিরতি;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            যদি (ok) {
                int64_t current_nelements = info.t.ne[0];
                জন্য (পূর্ণসংখ্যা i = 1; i < GGML_MAX_DIMS; ++i) {
                    যদি (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        যদি (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = মিথ্যা;
                             বিরতি;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 যদি (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     বিরতি;
                 }
            }
        }
        যদি (!ok) {
            বিরতি;
        }

        // tensor type
        {
            ok = ok && gr.পঠ(info.t.type);

            // check that tensor type is within defined range
            যদি (info.t.type < 0 || info.t.type >= GGML_TYPE_গুনতি) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_গুনতি);
                ok = মিথ্যা;
                বিরতি;
            }
            ধ্রুবক size_t  type_আকার = ggml_type_আকার(info.t.type);
            ধ্রুবক int64_t blck_আকার = ggml_blck_আকার(info.t.type);

            // check that row size is divisible by block size
            যদি (blck_আকার == 0 || info.t.ne[0] % blck_আকার != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (পূর্ণসংখ্যা) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_আকার);
                ok = মিথ্যা;
                বিরতি;
            }

            // check that the size of the tensor in bytes is representable
            যদি (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_আকার(info.t.type)) > SIZE_MAX/ggml_type_আকার(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = মিথ্যা;
                বিরতি;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_আকার;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_আকার);
            জন্য (পূর্ণসংখ্যা j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        যদি (!ok) {
            বিরতি;
        }

        // tensor data offset within buffer
        ok = ok && gr.পঠ(info.offset);

        ctx->info.push_back(info);
    }

    যদি (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        ফেরত nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    যদি (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        ফেরত nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        জন্য (size_t i = 0; i < ctx->info.size(); ++i) {
            ধ্রুবক gguf_tensor_info & ti = ctx->info[i];
            যদি (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                ফেরত nullptr;
            }
            ধ্রুবক size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            যদি (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                ফেরত nullptr;
            }
            ধ্রুবক size_t padded_আকার = GGML_PAD(nbytes, ctx->alignment);
            যদি (SIZE_MAX - ctx->size < padded_আকার) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_আকার);
                gguf_free(ctx);
                ফেরত nullptr;
            }
            ctx->size += padded_আকার;
        }
    }

    // load the tensor data only if requested
    যদি (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_আকার = 0;
        যদি (params.no_alloc) {
            যদি (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ফেরত nullptr;
            }

            ধ্রুবক size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_আকার = overhead;
        } অন্যথা {
            যদি ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ফেরত nullptr;
            }

            ধ্রুবক size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            যদি (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ফেরত nullptr;
            }

            mem_আকার = overhead + ctx->size;
        }

        কাঠামো ggml_init_params pdata = {
            /*mem_size   =*/ mem_আকার,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        যদি (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            ফেরত nullptr;
        }

        কাঠামো ggml_context * ctx_তথ্য = *params.ctx;

        কাঠামো ggml_tensor * তথ্য = nullptr;

        যদি (!params.no_alloc) {
            তথ্য = ggml_new_tensor_1d(ctx_তথ্য, GGML_TYPE_I8, ctx->size);

            ok = ok && তথ্য != nullptr;

            যদি (ok) {
                ggml_set_name(তথ্য, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.পঠ(তথ্য->তথ্য, ctx->size);

            যদি (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_তথ্য);
                *params.ctx = nullptr;
                gguf_free(ctx);
                ফেরত nullptr;
            }

            ctx->তথ্য = তথ্য->তথ্য;
        }

        ggml_set_no_alloc(ctx_তথ্য, সত্য);

        // create the tensors
        জন্য (size_t i = 0; i < ctx->info.size(); ++i) {
            ধ্রুবক কাঠামো gguf_tensor_info & info = ctx->info[i];

            কাঠামো ggml_tensor * cur = ggml_new_tensor(ctx_তথ্য, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            যদি (!ok) {
                বিরতি;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            যদি (!params.no_alloc) {
                cur->তথ্য = (অক্ষর *) তথ্য->তথ্য + info.offset;
            }
        }

        যদি (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_তথ্য);
            *params.ctx = nullptr;
            gguf_free(ctx);
            ফেরত nullptr;
        }

        ggml_set_no_alloc(ctx_তথ্য, params.no_alloc);
    }

    ফেরত ctx;
}

কাঠামো gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, শূন্য * userdata, size_t max_chunk_read, uint64_t max_expected_আকার, কাঠামো gguf_init_params params) {
    যদি (callback == nullptr) {
        ফেরত nullptr;
    }

    ধ্রুবক কাঠামো gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_আকার);
    ফেরত gguf_init_from_reader(gr, params);
}

কাঠামো gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

স্থির size_t gguf_file_reader_callback(শূন্য * userdata, শূন্য * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    যদি (reader.offset != offset) {
        যদি (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            ফেরত 0;
        }

        reader.offset = offset;
    }

    ধ্রুবক size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    ফেরত nread;
}

কাঠামো gguf_context * gguf_init_from_file_ptr(FILE * file, কাঠামো gguf_init_params params) {
    যদি (!file) {
        ফেরত nullptr;
    }

    ধ্রুবক int64_t cur = gguf_ftell(file);
    যদি (cur < 0) {
        ফেরত nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ধ্রুবক কাঠামো gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    ফেরত gguf_init_from_reader(gr, params);
}

কাঠামো gguf_বাফার_reader {
    ধ্রুবক uint8_t * তথ্য;
    size_t          size;
};

স্থির size_t gguf_বাফার_reader_callback(শূন্য * userdata, শূন্য * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    ধ্রুবক gguf_বাফার_reader & reader = *static_cast<gguf_বাফার_reader *>(userdata);

    যদি (offset > reader.size || len > reader.size - offset) {
        ফেরত 0;
    }

    ধ্রুবক size_t তথ্য_offset = static_cast<size_t>(offset);
    ধ্রুবক size_t nread = std::min(len, reader.size - তথ্য_offset);
    memcpy(static_cast<uint8_t *>(output), reader.তথ্য + তথ্য_offset, nread);
    ফেরত nread;
}

কাঠামো gguf_context * gguf_init_from_বাফার(ধ্রুবক শূন্য * তথ্য, size_t size, কাঠামো gguf_init_params params) {
    যদি (তথ্য == nullptr || size == 0) {
        ফেরত nullptr;
    }

    gguf_বাফার_reader reader = {
        /*.data = */ static_cast<ধ্রুবক uint8_t *>(তথ্য),
        /*.size = */ size,
    };
    ধ্রুবক কাঠামো gguf_reader gr(gguf_বাফার_reader_callback, &reader, SIZE_MAX, 0, size);
    ফেরত gguf_init_from_reader(gr, params);
}

কাঠামো gguf_context * gguf_init_from_file(ধ্রুবক অক্ষর * fname, কাঠামো gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    যদি (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        ফেরত nullptr;
    }

    কাঠামো gguf_context * ফলাফল = gguf_init_from_file_ptr(file, params);
    fclose(file);
    ফেরত ফলাফল;
}

শূন্য gguf_free(কাঠামো gguf_context * ctx) {
    যদি (ctx == nullptr) {
        ফেরত;
    }
    delete ctx;
}

ধ্রুবক অক্ষর * gguf_type_name(গণনা gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    ফেরত it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(ধ্রুবক কাঠামো gguf_context * ctx) {
    ফেরত ctx->version;
}

size_t gguf_get_alignment(ধ্রুবক কাঠামো gguf_context * ctx) {
    ফেরত ctx->alignment;
}

size_t gguf_get_তথ্য_offset(ধ্রুবক কাঠামো gguf_context * ctx) {
    ফেরত ctx->offset;
}

int64_t gguf_get_n_kv(ধ্রুবক কাঠামো gguf_context * ctx) {
    ফেরত ctx->kv.size();
}

int64_t gguf_find_চাবি(ধ্রুবক কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ধ্রুবক int64_t n_kv = gguf_get_n_kv(ctx);

    জন্য (int64_t i = 0; i < n_kv; ++i) {
        যদি (strcmp(চাবি, gguf_get_চাবি(ctx, i)) == 0) {
            keyfound = i;
            বিরতি;
        }
    }

    ফেরত keyfound;
}

ধ্রুবক অক্ষর * gguf_get_চাবি(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    ফেরত ctx->kv[চাবি_id].get_চাবি().c_str();
}

গণনা gguf_type gguf_get_kv_type(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    ফেরত ctx->kv[চাবি_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[চাবি_id].get_type();
}

গণনা gguf_type gguf_get_arr_type(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].is_array);
    ফেরত ctx->kv[চাবি_id].get_type();
}

ধ্রুবক শূন্য * gguf_get_arr_তথ্য(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_type() != GGUF_TYPE_STRING);
    ফেরত ctx->kv[চাবি_id].তথ্য.তথ্য();
}

ধ্রুবক অক্ষর * gguf_get_arr_str(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id, size_t i) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_type() == GGUF_TYPE_STRING);
    ফেরত ctx->kv[চাবি_id].তথ্য_string[i].c_str();
}

size_t gguf_get_arr_n(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));

    যদি (ctx->kv[চাবি_id].type == GGUF_TYPE_STRING) {
        ফেরত ctx->kv[চাবি_id].তথ্য_string.size();
    }

    ধ্রুবক size_t type_আকার = gguf_type_আকার(ctx->kv[চাবি_id].type);
    GGML_ASSERT(ctx->kv[চাবি_id].তথ্য.size() % type_আকার == 0);
    ফেরত ctx->kv[চাবি_id].তথ্য.size() / type_আকার;
}

uint8_t gguf_get_val_u8(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<int32_t>();
}

দশমিক gguf_get_val_f32(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<দশমিক>();
}

uint64_t gguf_get_val_u64(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<int64_t>();
}

দ্বিগুণ gguf_get_val_f64(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<দ্বিগুণ>();
}

যৌক্তিক gguf_get_val_bool(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<যৌক্তিক>();
}

ধ্রুবক অক্ষর * gguf_get_val_str(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    ফেরত ctx->kv[চাবি_id].get_val<std::string>().c_str();
}

ধ্রুবক শূন্য * gguf_get_val_তথ্য(ধ্রুবক কাঠামো gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[চাবি_id].get_type() != GGUF_TYPE_STRING);
    ফেরত ctx->kv[চাবি_id].তথ্য.তথ্য();
}

int64_t gguf_get_n_tensors(ধ্রুবক কাঠামো gguf_context * ctx) {
    ফেরত ctx->info.size();
}

int64_t gguf_find_tensor(ধ্রুবক কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ধ্রুবক int64_t n_tensors = gguf_get_n_tensors(ctx);

    জন্য (int64_t i = 0; i < n_tensors; ++i) {
        যদি (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            বিরতি;
        }
    }

    ফেরত tensor_id;
}

size_t gguf_get_tensor_offset(ধ্রুবক কাঠামো gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ফেরত ctx->info[tensor_id].offset;
}

ধ্রুবক অক্ষর * gguf_get_tensor_name(ধ্রুবক কাঠামো gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ফেরত ctx->info[tensor_id].t.name;
}

ধ্রুবক int64_t * gguf_get_tensor_ne(ধ্রুবক কাঠামো gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ফেরত ctx->info[tensor_id].t.ne;
}

গণনা ggml_type gguf_get_tensor_type(ধ্রুবক কাঠামো gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ফেরত ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_আকার(ধ্রুবক কাঠামো gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ফেরত ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_চাবি(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি) {
    ধ্রুবক int64_t চাবি_id = gguf_find_চাবি(ctx, চাবি);
    যদি (চাবি_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + চাবি_id);
    }
    ফেরত চাবি_id;
}

template<typename T>
স্থির শূন্য gguf_check_reserved_চাবিগুলি(ধ্রুবক std::string & চাবি, ধ্রুবক T val) {
    যদি (চাবি == GGUF_চাবি_GENERAL_ALIGNMENT) {
        যদি constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_চাবি_GENERAL_ALIGNMENT " must be power of 2");
        } অন্যথা {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_চাবি_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

শূন্য gguf_set_val_u8(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, uint8_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i8(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, int8_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_u16(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, uint16_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i16(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, int16_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_u32(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, uint32_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i32(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, int32_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_f32(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, দশমিক val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_u64(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, uint64_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i64(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, int64_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_f64(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, দ্বিগুণ val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_bool(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, যৌক্তিক val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_str(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, ধ্রুবক অক্ষর * val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, std::string(val));
}

শূন্য gguf_set_arr_তথ্য(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, গণনা gguf_type type, ধ্রুবক শূন্য * তথ্য, size_t n) {
    gguf_check_reserved_চাবিগুলি(চাবি, তথ্য);
    gguf_remove_চাবি(ctx, চাবি);

    ধ্রুবক size_t nbytes = n*gguf_type_আকার(type);
    std::vector<int8_t> tmp(nbytes);
    যদি (!tmp.empty()) {
        memcpy(tmp.তথ্য(), তথ্য, nbytes);
    }
    ctx->kv.emplace_back(চাবি, tmp);
    ctx->kv.back().cast(type);
}

শূন্য gguf_set_arr_str(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * চাবি, ধ্রুবক অক্ষর ** তথ্য, size_t n) {
    gguf_check_reserved_চাবিগুলি(চাবি, তথ্য);
    gguf_remove_চাবি(ctx, চাবি);

    std::vector<std::string> tmp(n);
    জন্য (size_t i = 0; i < n; ++i) {
        tmp[i] = তথ্য[i];
    }
    ctx->kv.emplace_back(চাবি, tmp);
}

// set or add KV pairs from another context
শূন্য gguf_set_kv(কাঠামো gguf_context * ctx, ধ্রুবক কাঠামো gguf_context * src) {
    ধ্রুবক int64_t n_kv = gguf_get_n_kv(src);
    জন্য (int64_t i = 0; i < n_kv; ++i) {
        ধ্রুবক কাঠামো gguf_kv & kv = src->kv[i];

        যদি (!kv.is_array) {
            নির্বাচন (kv.get_type()) {
                ক্ষেত্র GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_চাবি().c_str(), kv.get_val<uint8_t>());             বিরতি;
                ক্ষেত্র GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_চাবি().c_str(), kv.get_val<int8_t>());              বিরতি;
                ক্ষেত্র GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_চাবি().c_str(), kv.get_val<uint16_t>());            বিরতি;
                ক্ষেত্র GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_চাবি().c_str(), kv.get_val<int16_t>());             বিরতি;
                ক্ষেত্র GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_চাবি().c_str(), kv.get_val<uint32_t>());            বিরতি;
                ক্ষেত্র GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_চাবি().c_str(), kv.get_val<int32_t>());             বিরতি;
                ক্ষেত্র GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_চাবি().c_str(), kv.get_val<দশমিক>());               বিরতি;
                ক্ষেত্র GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_চাবি().c_str(), kv.get_val<uint64_t>());            বিরতি;
                ক্ষেত্র GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_চাবি().c_str(), kv.get_val<int64_t>());             বিরতি;
                ক্ষেত্র GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_চাবি().c_str(), kv.get_val<দ্বিগুণ>());              বিরতি;
                ক্ষেত্র GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_চাবি().c_str(), kv.get_val<যৌক্তিক>());                বিরতি;
                ক্ষেত্র GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_চাবি().c_str(), kv.get_val<std::string>().c_str()); বিরতি;
                ক্ষেত্র GGUF_TYPE_ARRAY:
                পূর্বনির্ধারিত: GGML_ABORT("invalid type");
            }
            চালু;
        }

        ধ্রুবক size_t ne = kv.get_ne();

        নির্বাচন (kv.get_type()) {
            ক্ষেত্র GGUF_TYPE_UINT8:
            ক্ষেত্র GGUF_TYPE_INT8:
            ক্ষেত্র GGUF_TYPE_UINT16:
            ক্ষেত্র GGUF_TYPE_INT16:
            ক্ষেত্র GGUF_TYPE_UINT32:
            ক্ষেত্র GGUF_TYPE_INT32:
            ক্ষেত্র GGUF_TYPE_FLOAT32:
            ক্ষেত্র GGUF_TYPE_UINT64:
            ক্ষেত্র GGUF_TYPE_INT64:
            ক্ষেত্র GGUF_TYPE_FLOAT64:
            ক্ষেত্র GGUF_TYPE_BOOL: {
                gguf_set_arr_তথ্য(ctx, kv.get_চাবি().c_str(), kv.get_type(), kv.তথ্য.তথ্য(), ne);
            } বিরতি;
            ক্ষেত্র GGUF_TYPE_STRING: {
                std::vector<ধ্রুবক অক্ষর *> tmp(ne);
                জন্য (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.তথ্য_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_চাবি().c_str(), tmp.তথ্য(), ne);
            } বিরতি;
            ক্ষেত্র GGUF_TYPE_ARRAY:
            পূর্বনির্ধারিত: GGML_ABORT("invalid type");
        }
    }
}

শূন্য gguf_add_tensor(
             কাঠামো gguf_context * ctx,
        ধ্রুবক কাঠামো ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    যদি (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    কাঠামো gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

শূন্য gguf_set_tensor_type(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * name, গণনা ggml_type type) {
    ধ্রুবক int64_t tensor_id = gguf_find_tensor(ctx, name);
    যদি (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    কাঠামো ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ধ্রুবক size_t  type_আকার = ggml_type_আকার(type);
    ধ্রুবক int64_t blck_আকার = ggml_blck_আকার(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_আকার == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_আকার;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_আকার);
    জন্য (পূর্ণসংখ্যা i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ধ্রুবক int64_t n_tensors = gguf_get_n_tensors(ctx);
    জন্য (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

শূন্য gguf_set_tensor_তথ্য(কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * name, ধ্রুবক শূন্য * তথ্য) {
    ধ্রুবক int64_t tensor_id = gguf_find_tensor(ctx, name);
    যদি (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.তথ্য = (শূন্য *)(uintptr_t)তথ্য; // double cast suppresses warning about casting away const
}

কাঠামো gguf_writer_base {
    size_t written_bytes {0u};

    আভাসী ~gguf_writer_base(শূন্য) = পূর্বনির্ধারিত;

    // we bet on devirtualization
    আভাসী শূন্য লিখ(int8_t val) = 0;
    আভাসী শূন্য লিখ(ধ্রুবক std::vector<int8_t> & val) = 0;
    আভাসী শূন্য write_tensor_তথ্য(ধ্রুবক কাঠামো gguf_tensor_info & info, size_t offset_তথ্য, size_t alignment) = 0;

    template <typename T>
    শূন্য লিখ(ধ্রুবক T & val) {
        জন্য (size_t i = 0; i < আকার(val); ++i) {
            লিখ(reinterpret_cast<ধ্রুবক int8_t *>(&val)[i]);
        }
    }

    শূন্য লিখ(ধ্রুবক যৌক্তিক & val) {
        ধ্রুবক int8_t val8 = val ? 1 : 0;
        লিখ(val8);
    }

    শূন্য লিখ(ধ্রুবক std::string & val) {
        {
            ধ্রুবক uint64_t n = val.দৈর্ঘ্য();
            লিখ(n);
        }
        জন্য (size_t i = 0; i < val.দৈর্ঘ্য(); ++i) {
            লিখ((val.তথ্য())[i]);
        }
    }

    শূন্য লিখ(ধ্রুবক অক্ষর * val) {
        লিখ(std::string(val));
    }

    শূন্য লিখ(ধ্রুবক গণনা ggml_type & val) {
        লিখ(int32_t(val));
    }

    শূন্য লিখ(ধ্রুবক গণনা gguf_type & val) {
        লিখ(int32_t(val));
    }

    শূন্য লিখ(ধ্রুবক কাঠামো gguf_kv & kv) {
        ধ্রুবক uint64_t ne = kv.get_ne();

        লিখ(kv.get_চাবি());

        যদি (kv.is_array) {
            লিখ(GGUF_TYPE_ARRAY);
            লিখ(kv.get_type());
            লিখ(ne);
        } অন্যথা {
            লিখ(kv.get_type());
        }

        নির্বাচন (kv.get_type()) {
            ক্ষেত্র GGUF_TYPE_UINT8:
            ক্ষেত্র GGUF_TYPE_INT8:
            ক্ষেত্র GGUF_TYPE_UINT16:
            ক্ষেত্র GGUF_TYPE_INT16:
            ক্ষেত্র GGUF_TYPE_UINT32:
            ক্ষেত্র GGUF_TYPE_INT32:
            ক্ষেত্র GGUF_TYPE_FLOAT32:
            ক্ষেত্র GGUF_TYPE_UINT64:
            ক্ষেত্র GGUF_TYPE_INT64:
            ক্ষেত্র GGUF_TYPE_FLOAT64: {
                লিখ(kv.তথ্য);
            } বিরতি;
            ক্ষেত্র GGUF_TYPE_BOOL: {
                জন্য (size_t i = 0; i < ne; ++i) {
                    লিখ(kv.get_val<যৌক্তিক>(i));
                }
            } বিরতি;
            ক্ষেত্র GGUF_TYPE_STRING: {
                জন্য (size_t i = 0; i < ne; ++i) {
                    লিখ(kv.get_val<std::string>(i));
                }
            } বিরতি;
            ক্ষেত্র GGUF_TYPE_ARRAY:
            পূর্বনির্ধারিত: GGML_ABORT("invalid type");
        }
    }

    শূন্য write_tensor_meta(ধ্রুবক কাঠামো gguf_tensor_info & info) {
        লিখ(info.t.name);

        ধ্রুবক uint32_t n_dims = ggml_n_dims(&info.t);
        লিখ(n_dims);

        জন্য (uint32_t j = 0; j < n_dims; ++j) {
            লিখ(info.t.ne[j]);
        }
        লিখ(info.t.type);
        লিখ(info.offset);
    }

    শূন্য pad(ধ্রুবক size_t alignment) {
        যখন (written_bytes % alignment != 0) {
            ধ্রুবক int8_t zero = 0;
            লিখ(zero);
        }
    }
};

// vector buffer based writer
কাঠামো gguf_writer_buf final : সর্বজনীন gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::লিখ;

    শূন্য লিখ(ধ্রুবক int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    শূন্য লিখ(ধ্রুবক std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    শূন্য write_tensor_তথ্য(ধ্রুবক কাঠামো gguf_tensor_info & info, ধ্রুবক size_t offset_তথ্য, ধ্রুবক size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_তথ্য == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ধ্রুবক size_t offset = buf.size();
        ধ্রুবক size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        যদি (info.t.বাফার) {
            ggml_backend_tensor_get(&info.t, buf.তথ্য() + offset, 0, nbytes);
        } অন্যথা {
            GGML_ASSERT(info.t.তথ্য);
            memcpy(buf.তথ্য() + offset, info.t.তথ্য, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
কাঠামো gguf_writer_file final : সর্বজনীন gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::লিখ;

    শূন্য লিখ(ধ্রুবক int8_t val) override {
        ধ্রুবক auto real_val = static_cast<uint8_t>(val);
        ধ্রুবক auto ret = fputc(real_val, file);
        written_bytes++;
        যদি (ret != real_val) {
            ছোড়ো std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((পূর্ণসংখ্যা)real_val) + "'");
        }
    }

    শূন্য লিখ(ধ্রুবক std::vector<int8_t> & val) override {
        ধ্রুবক auto ret = fwrite(val.তথ্য(), 1, val.size(), file);
        written_bytes += val.size();
        যদি (ret != val.size()) {
            ছোড়ো std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    শূন্য write_tensor_তথ্য(ধ্রুবক কাঠামো gguf_tensor_info & info, ধ্রুবক size_t offset_তথ্য, ধ্রুবক size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_তথ্য == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ধ্রুবক size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        যদি (info.t.বাফার) {
            ggml_backend_tensor_get(&info.t, buf.তথ্য(), 0, nbytes);
        } অন্যথা {
            GGML_ASSERT(info.t.তথ্য);
            memcpy(buf.তথ্য(), info.t.তথ্য, nbytes);
        }
        লিখ(buf);

        pad(alignment);
    }
};

template <typename writer_t>
স্থির শূন্য gguf_write_out(ধ্রুবক কাঠামো gguf_context * ctx, writer_t & gw, যৌক্তিক only_meta) {
    ধ্রুবক int64_t n_kv      = gguf_get_n_kv(ctx);
    ধ্রুবক int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.লিখ(GGUF_MAGIC[0]);
    gw.লিখ(GGUF_MAGIC[1]);
    gw.লিখ(GGUF_MAGIC[2]);
    gw.লিখ(GGUF_MAGIC[3]);
    gw.লিখ(ctx->version);
    gw.লিখ(n_tensors);
    gw.লিখ(n_kv);

    // write key-value pairs
    জন্য (int64_t i = 0; i < n_kv; ++i) {
        gw.লিখ(ctx->kv[i]);
    }

    // write tensor info
    জন্য (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    যদি (only_meta) {
        ফেরত;
    }

    ধ্রুবক size_t offset_তথ্য = gw.written_bytes;

    // write tensor data
    জন্য (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_তথ্য(ctx->info[i], offset_তথ্য, ctx->alignment);
    }
}

শূন্য gguf_write_to_buf(ধ্রুবক কাঠামো gguf_context * ctx, std::vector<int8_t> & buf, যৌক্তিক only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

যৌক্তিক gguf_write_to_file_ptr(ধ্রুবক কাঠামো gguf_context * ctx, FILE * file, যৌক্তিক only_meta) {
    GGML_ASSERT(file);

    চেষ্টা {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } ধরো (ধ্রুবক std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
}

যৌক্তিক gguf_write_to_file(ধ্রুবক কাঠামো gguf_context * ctx, ধ্রুবক অক্ষর * fname, যৌক্তিক only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    যদি (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        ফেরত মিথ্যা;
    }

    ধ্রুবক যৌক্তিক success = gguf_write_to_file_ptr(ctx, file, only_meta);
    যদি (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    ফেরত success;
}

size_t gguf_get_meta_আকার(ধ্রুবক কাঠামো gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ সত্য);
    ফেরত buf.size();
}

শূন্য gguf_get_meta_তথ্য(ধ্রুবক কাঠামো gguf_context * ctx, শূন্য * তথ্য) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ সত্য);
    memcpy(তথ্য, buf.তথ্য(), buf.size());
}
