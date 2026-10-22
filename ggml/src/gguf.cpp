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

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#অন্যথা
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
সংরচনা type_to_gguf_type;

template <>
সংরচনা type_to_gguf_type<uint8_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT8;
};

template <>
সংরচনা type_to_gguf_type<int8_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT8;
};

template <>
সংরচনা type_to_gguf_type<uint16_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT16;
};

template <>
সংরচনা type_to_gguf_type<int16_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT16;
};

template <>
সংরচনা type_to_gguf_type<uint32_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT32;
};

template <>
সংরচনা type_to_gguf_type<int32_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT32;
};

template <>
সংরচনা type_to_gguf_type<ভিন্ন> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
সংরচনা type_to_gguf_type<তার্কিক> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_BOOL;
};

template <>
সংরচনা type_to_gguf_type<std::string> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_STRING;
};

template <>
সংরচনা type_to_gguf_type<uint64_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_UINT64;
};

template <>
সংরচনা type_to_gguf_type<int64_t> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_INT64;
};

template <>
সংরচনা type_to_gguf_type<দ্বিগুণ> {
    স্থির constexpr গণনা gguf_type value = GGUF_TYPE_FLOAT64;
};

স্থির ধ্রুব std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   পরিমাণ(uint8_t)},
    {GGUF_TYPE_INT8,    পরিমাণ(int8_t)},
    {GGUF_TYPE_UINT16,  পরিমাণ(uint16_t)},
    {GGUF_TYPE_INT16,   পরিমাণ(int16_t)},
    {GGUF_TYPE_UINT32,  পরিমাণ(uint32_t)},
    {GGUF_TYPE_INT32,   পরিমাণ(int32_t)},
    {GGUF_TYPE_FLOAT32, পরিমাণ(ভিন্ন)},
    {GGUF_TYPE_BOOL,    পরিমাণ(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  পরিমাণ(uint64_t)},
    {GGUF_TYPE_INT64,   পরিমাণ(int64_t)},
    {GGUF_TYPE_FLOAT64, পরিমাণ(দ্বিগুণ)},
};
static_assert(GGUF_TYPE_গুনতি == 13, "GGUF_TYPE_COUNT != 13");

স্থির ধ্রুব std::map<gguf_type, ধ্রুব বর্ণ *> GGUF_TYPE_NAME = {
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

size_t gguf_type_size(গণনা gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    প্রত্যাগম it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

সংরচনা gguf_kv {
    std::string চাবি;

    তার্কিক is_array;
    গণনা gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(ধ্রুব std::string & চাবি, ধ্রুব T value)
            : চাবি(চাবি), is_array(অসত্য), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!চাবি.empty());
        data.resize(পরিমাণ(T));
        memcpy(data.data(), &value, পরিমাণ(T));
    }

    template <typename T>
    gguf_kv(ধ্রুব std::string & চাবি, ধ্রুব std::vector<T> & value)
            : চাবি(চাবি), is_array(সত্য), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!চাবি.empty());
        data.resize(value.size()*পরিমাণ(T));
        প্রতি (size_t i = 0; i < value.size(); ++i) {
            ধ্রুব T tmp = value[i];
            memcpy(data.data() + i*পরিমাণ(T), &tmp, পরিমাণ(T));
        }
    }

    gguf_kv(ধ্রুব std::string & চাবি, ধ্রুব std::string & value)
            : চাবি(চাবি), is_array(অসত্য), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!চাবি.empty());
        data_string.push_back(value);
    }

    gguf_kv(ধ্রুব std::string & চাবি, ধ্রুব std::vector<std::string> & value)
            : চাবি(চাবি), is_array(সত্য), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!চাবি.empty());
        data_string = value;
    }

    ধ্রুব std::string & get_চাবি() ধ্রুব {
        প্রত্যাগম চাবি;
    }

    ধ্রুব গণনা gguf_type & get_type() ধ্রুব {
        প্রত্যাগম type;
    }

    size_t get_ne() ধ্রুব {
        যদি (type == GGUF_TYPE_STRING) {
            ধ্রুব size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            প্রত্যাগম ne;
        }
        ধ্রুব size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        ধ্রুব size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        প্রত্যাগম ne;
    }

    template <typename T>
    ধ্রুব T & get_val(ধ্রুব size_t i = 0) ধ্রুব {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        যদি constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            প্রত্যাগম data_string[i];
        }
        ধ্রুব size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        প্রত্যাগম reinterpret_cast<ধ্রুব T *>(data.data())[i];
    }

    শূন্য cast(ধ্রুব গণনা gguf_type new_type) {
        ধ্রুব size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

সংরচনা gguf_tensor_info {
    সংরচনা ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

সংরচনা gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<সংরচনা gguf_kv> kv;
    std::vector<সংরচনা gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    শূন্য * data = nullptr;
};

সংরচনা gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            শূন্য * userdata,
            size_t max_chunk_read,
            uint64_t data_offset = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          start_offset(data_offset),
          data_offset(data_offset),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    স্থির uint64_t file_remain(FILE * file) {
        ধ্রুব int64_t cur = gguf_ftell(file);
        যদি (cur < 0) {
            প্রত্যাগম 0;
        }
        যদি (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            প্রত্যাগম 0;
        }
        ধ্রুব int64_t end = gguf_ftell(file);
        যদি (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            প্রত্যাগম 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        প্রত্যাগম static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    তার্কিক পঠ(T & dst) ধ্রুব {
        ধ্রুব size_t size = পরিমাণ(dst);
        যদি (size > nbytes_remain) {
            প্রত্যাগম অসত্য;
        }
        প্রত্যাগম read_raw(&dst, size) == size;
    }

    template <typename T>
    তার্কিক পঠ(std::vector<T> & dst, ধ্রুব size_t n) ধ্রুব {
        যদি (n > GGUF_MAX_ARRAY_ELEMENTS) {
            প্রত্যাগম অসত্য;
        }
        যদি constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            যদি (n > SIZE_MAX / পরিমাণ(uint64_t)) {
                প্রত্যাগম অসত্য;
            }
            যদি (nbytes_remain < n * পরিমাণ(uint64_t)) {
                প্রত্যাগম অসত্য;
            }
        } অন্যথা {
            যদি (n > SIZE_MAX / পরিমাণ(T)) {
                প্রত্যাগম অসত্য;
            }
            যদি (nbytes_remain < n * পরিমাণ(T)) {
                প্রত্যাগম অসত্য;
            }
        }
        dst.resize(n);
        প্রতি (size_t i = 0; i < dst.size(); ++i) {
            যদি constexpr (std::is_same<T, তার্কিক>::value) {
                তার্কিক tmp;
                যদি (!পঠ(tmp)) {
                    প্রত্যাগম অসত্য;
                }
                dst[i] = tmp;
            } অন্যথা {
                যদি (!পঠ(dst[i])) {
                    প্রত্যাগম অসত্য;
                }
            }
        }
        প্রত্যাগম সত্য;
    }

    তার্কিক পঠ(তার্কিক & dst) ধ্রুব {
        int8_t tmp = -1;
        যদি (!পঠ(tmp)) {
            প্রত্যাগম অসত্য;
        }
        dst = tmp != 0;
        প্রত্যাগম সত্য;
    }

    তার্কিক পঠ(গণনা ggml_type & dst) ধ্রুব {
        int32_t tmp = -1;
        যদি (!পঠ(tmp)) {
            প্রত্যাগম অসত্য;
        }
        dst = ggml_type(tmp);
        প্রত্যাগম সত্য;
    }

    তার্কিক পঠ(গণনা gguf_type & dst) ধ্রুব {
        int32_t tmp = -1;
        যদি (!পঠ(tmp)) {
            প্রত্যাগম অসত্য;
        }
        dst = gguf_type(tmp);
        প্রত্যাগম সত্য;
    }

    তার্কিক পঠ(std::string & dst) ধ্রুব {
        uint64_t size = 0;
        যদি (!পঠ(size)) {
            প্রত্যাগম অসত্য;
        }
        যদি (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            প্রত্যাগম অসত্য;
        }
        যদি (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            প্রত্যাগম অসত্য;
        }
        dst.resize(static_cast<size_t>(size));
        প্রত্যাগম read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    তার্কিক পঠ(শূন্য * dst, ধ্রুব size_t size) ধ্রুব {
        যদি (size > nbytes_remain) {
            প্রত্যাগম অসত্য;
        }
        প্রত্যাগম read_raw(dst, size) == size;
    }

    uint64_t tell() ধ্রুব {
        প্রত্যাগম data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ধ্রুব {
        প্রত্যাগম start_offset;
    }

    তার্কিক seek(uint64_t absolute_offset) ধ্রুব {
        ধ্রুব uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        যদি (absolute_offset > end_offset) {
            প্রত্যাগম অসত্য;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        প্রত্যাগম সত্য;
    }

গুপ্ত:
    size_t read_raw(শূন্য * dst, size_t size) ধ্রুব {
        যদি (callback == nullptr || size == 0) {
            প্রত্যাগম 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        তার্কিক reached_eof = অসত্য;

        যাবত্ (total_nread < size) {
            ধ্রুব size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            যদি (data_offset + total_nread < data_offset) {
                ভঙ্গ;
            }
            ধ্রুব size_t nread = callback(userdata, static_cast<শূন্য *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            যদি (nread != chunk_size) {
                reached_eof = সত্য;
                ভঙ্গ;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        যদি (reached_eof) {
            nbytes_remain = 0;
        }

        প্রত্যাগম total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    শূন্য * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

সংরচনা gguf_context * gguf_init_empty(শূন্য) {
    প্রত্যাগম নূতন gguf_context;
}

template<typename T>
তার্কিক gguf_read_emplace_helper(ধ্রুব সংরচনা gguf_reader & gr, std::vector<সংরচনা gguf_kv> & kv, ধ্রুব std::string & চাবি, ধ্রুব তার্কিক is_array, ধ্রুব size_t n) {
    যদি (is_array) {
        std::vector<T> value;
        যত্ন {
            যদি (!gr.পঠ(value, n)) {
                প্রত্যাগম অসত্য;
            }
        } গৃহাণ (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, চাবি.c_str());
            প্রত্যাগম অসত্য;
        } গৃহাণ (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, চাবি.c_str());
            প্রত্যাগম অসত্য;
        }
        kv.emplace_back(চাবি, value);
    } অন্যথা {
        T value;
        যদি (!gr.পঠ(value)) {
            প্রত্যাগম অসত্য;
        }
        kv.emplace_back(চাবি, value);
    }
    প্রত্যাগম সত্য;
}

স্থির সংরচনা gguf_context * gguf_init_from_reader(ধ্রুব সংরচনা gguf_reader & gr, সংরচনা gguf_init_params params) {
    সংরচনা gguf_context * ctx = নূতন gguf_context;

    তার্কিক ok = সত্য;

    // file magic
    {
        std::vector<বর্ণ> magic;
        ok = ok && gr.পঠ(magic, 4);

        যদি (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            প্রত্যাগম nullptr;
        }

        প্রতি (uint32_t i = 0; i < magic.size(); i++) {
            যদি (magic[i] != GGUF_MAGIC[i]) {
                বর্ণ c0 = isprint(magic[0]) ? magic[0] : '?';
                বর্ণ c1 = isprint(magic[1]) ? magic[1] : '?';
                বর্ণ c2 = isprint(magic[2]) ? magic[2] : '?';
                বর্ণ c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    যদি (ok && gr.পঠ(ctx->version)) {
        যদি (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = অসত্য;
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
            ok = অসত্য;
        }

        যদি (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = অসত্য;
        }
        যদি (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = অসত্য;
        }
    } অন্যথা {
        ok = অসত্য;
    }

    যদি (ok && gr.পঠ(n_tensors)) {
        static_assert(পরিমাণ(size_t) <= 8 && পরিমাণ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        যদি (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/পরিমাণ(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/পরিমাণ(gguf_tensor_info));
            ok = অসত্য;
        }
    } অন্যথা {
        ok = অসত্য;
    }

    যদি (ok && gr.পঠ(n_kv)) {
        static_assert(পরিমাণ(size_t) <= 8 && পরিমাণ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        যদি (n_kv < 0 || n_kv > int64_t(SIZE_MAX/পরিমাণ(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/পরিমাণ(gguf_kv));
            ok = অসত্য;
        }
    } অন্যথা {
        ok = অসত্য;
    }

    যদি (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        প্রত্যাগম nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_চাবিগুলি;

        প্রতি (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string চাবি;
            gguf_type   type     = gguf_type(-1);
            তার্কিক        is_array = অসত্য;
            uint64_t    n        = 1;

            যত্ন {
                ok = ok && gr.পঠ(চাবি);
            } গৃহাণ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = অসত্য;
            } গৃহাণ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = অসত্য;
            }
            যদি (ok && চাবি.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = অসত্য;
            }
            যদি (ok && !seen_চাবিগুলি.insert(চাবি).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, চাবি.c_str(), i);
                ok = অসত্য;
            }
            যদি (!ok) {
                ভঙ্গ;
            }

            ok = ok && gr.পঠ(type);
            যদি (type == GGUF_TYPE_ARRAY) {
                is_array = সত্য;
                ok = ok && gr.পঠ(type);
                ok = ok && gr.পঠ(n);
            }
            যদি (!ok) {
                ভঙ্গ;
            }

            বিকল্প (type) {
                পক্ষ GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<ভিন্ন>      (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<তার্কিক>       (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<দ্বিগুণ>     (gr, ctx->kv, চাবি, is_array, n); ভঙ্গ;
                পক্ষ GGUF_TYPE_ARRAY:
                স্বভাব:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, চাবি.c_str(), type);
                        ok = অসত্য;
                    } ভঙ্গ;
            }
        }

        যদি (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            প্রত্যাগম nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ধ্রুব পূর্ণাঙ্ক alignment_idx = gguf_find_চাবি(ctx, GGUF_চাবি_GENERAL_ALIGNMENT);
        যদি (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_চাবি_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            প্রত্যাগম nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        যদি (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            প্রত্যাগম nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    প্রতি (int64_t i = 0; ok && i < n_tensors; ++i) {
        সংরচনা gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            যত্ন {
                ok = ok && gr.পঠ(name);
            } গৃহাণ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = অসত্য;
            } গৃহাণ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = অসত্য;
            }
            যদি (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = অসত্য;
                ভঙ্গ;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            যদি (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = অসত্য;
                ভঙ্গ;
            }
        }
        যদি (!ok) {
            ভঙ্গ;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.পঠ(n_dims);
            যদি (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = অসত্য;
                ভঙ্গ;
            }
            প্রতি (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                যদি (j < n_dims) {
                    ok = ok && gr.পঠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                যদি (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = অসত্য;
                    ভঙ্গ;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            যদি (ok) {
                int64_t current_nelements = info.t.ne[0];
                প্রতি (পূর্ণাঙ্ক i = 1; i < GGML_MAX_DIMS; ++i) {
                    যদি (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        যদি (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = অসত্য;
                             ভঙ্গ;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 যদি (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     ভঙ্গ;
                 }
            }
        }
        যদি (!ok) {
            ভঙ্গ;
        }

        // tensor type
        {
            ok = ok && gr.পঠ(info.t.type);

            // check that tensor type is within defined range
            যদি (info.t.type < 0 || info.t.type >= GGML_TYPE_গুনতি) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_গুনতি);
                ok = অসত্য;
                ভঙ্গ;
            }
            ধ্রুব size_t  type_size = ggml_type_size(info.t.type);
            ধ্রুব int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            যদি (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (পূর্ণাঙ্ক) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = অসত্য;
                ভঙ্গ;
            }

            // check that the size of the tensor in bytes is representable
            যদি (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = অসত্য;
                ভঙ্গ;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            প্রতি (পূর্ণাঙ্ক j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        যদি (!ok) {
            ভঙ্গ;
        }

        // tensor data offset within buffer
        ok = ok && gr.পঠ(info.offset);

        ctx->info.push_back(info);
    }

    যদি (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        প্রত্যাগম nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    যদি (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        প্রত্যাগম nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        প্রতি (size_t i = 0; i < ctx->info.size(); ++i) {
            ধ্রুব gguf_tensor_info & ti = ctx->info[i];
            যদি (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }
            ধ্রুব size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            যদি (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }
            ধ্রুব size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            যদি (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    যদি (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        যদি (params.no_alloc) {
            যদি (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }

            ধ্রুব size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } অন্যথা {
            যদি ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }

            ধ্রুব size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            যদি (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        সংরচনা ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        যদি (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            প্রত্যাগম nullptr;
        }

        সংরচনা ggml_context * ctx_data = *params.ctx;

        সংরচনা ggml_tensor * data = nullptr;

        যদি (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            যদি (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.পঠ(data->data, ctx->size);

            যদি (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                প্রত্যাগম nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, সত্য);

        // create the tensors
        প্রতি (size_t i = 0; i < ctx->info.size(); ++i) {
            ধ্রুব সংরচনা gguf_tensor_info & info = ctx->info[i];

            সংরচনা ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            যদি (!ok) {
                ভঙ্গ;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            যদি (!params.no_alloc) {
                cur->data = (বর্ণ *) data->data + info.offset;
            }
        }

        যদি (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            প্রত্যাগম nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    প্রত্যাগম ctx;
}

সংরচনা gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, শূন্য * userdata, size_t max_chunk_read, uint64_t max_expected_size, সংরচনা gguf_init_params params) {
    যদি (callback == nullptr) {
        প্রত্যাগম nullptr;
    }

    ধ্রুব সংরচনা gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    প্রত্যাগম gguf_init_from_reader(gr, params);
}

সংরচনা gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

স্থির size_t gguf_file_reader_callback(শূন্য * userdata, শূন্য * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    যদি (reader.offset != offset) {
        যদি (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            প্রত্যাগম 0;
        }

        reader.offset = offset;
    }

    ধ্রুব size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    প্রত্যাগম nread;
}

সংরচনা gguf_context * gguf_init_from_file_ptr(FILE * file, সংরচনা gguf_init_params params) {
    যদি (!file) {
        প্রত্যাগম nullptr;
    }

    ধ্রুব int64_t cur = gguf_ftell(file);
    যদি (cur < 0) {
        প্রত্যাগম nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ধ্রুব সংরচনা gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    প্রত্যাগম gguf_init_from_reader(gr, params);
}

সংরচনা gguf_buffer_reader {
    ধ্রুব uint8_t * data;
    size_t          size;
};

স্থির size_t gguf_buffer_reader_callback(শূন্য * userdata, শূন্য * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    ধ্রুব gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    যদি (offset > reader.size || len > reader.size - offset) {
        প্রত্যাগম 0;
    }

    ধ্রুব size_t data_offset = static_cast<size_t>(offset);
    ধ্রুব size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    প্রত্যাগম nread;
}

সংরচনা gguf_context * gguf_init_from_buffer(ধ্রুব শূন্য * data, size_t size, সংরচনা gguf_init_params params) {
    যদি (data == nullptr || size == 0) {
        প্রত্যাগম nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<ধ্রুব uint8_t *>(data),
        /*.size = */ size,
    };
    ধ্রুব সংরচনা gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    প্রত্যাগম gguf_init_from_reader(gr, params);
}

সংরচনা gguf_context * gguf_init_from_file(ধ্রুব বর্ণ * fname, সংরচনা gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    যদি (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        প্রত্যাগম nullptr;
    }

    সংরচনা gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    প্রত্যাগম result;
}

শূন্য gguf_free(সংরচনা gguf_context * ctx) {
    যদি (ctx == nullptr) {
        প্রত্যাগম;
    }
    delete ctx;
}

ধ্রুব বর্ণ * gguf_type_name(গণনা gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    প্রত্যাগম it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(ধ্রুব সংরচনা gguf_context * ctx) {
    প্রত্যাগম ctx->version;
}

size_t gguf_get_alignment(ধ্রুব সংরচনা gguf_context * ctx) {
    প্রত্যাগম ctx->alignment;
}

size_t gguf_get_data_offset(ধ্রুব সংরচনা gguf_context * ctx) {
    প্রত্যাগম ctx->offset;
}

int64_t gguf_get_n_kv(ধ্রুব সংরচনা gguf_context * ctx) {
    প্রত্যাগম ctx->kv.size();
}

int64_t gguf_find_চাবি(ধ্রুব সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ধ্রুব int64_t n_kv = gguf_get_n_kv(ctx);

    প্রতি (int64_t i = 0; i < n_kv; ++i) {
        যদি (strcmp(চাবি, gguf_get_চাবি(ctx, i)) == 0) {
            keyfound = i;
            ভঙ্গ;
        }
    }

    প্রত্যাগম keyfound;
}

ধ্রুব বর্ণ * gguf_get_চাবি(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    প্রত্যাগম ctx->kv[চাবি_id].get_চাবি().c_str();
}

গণনা gguf_type gguf_get_kv_type(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    প্রত্যাগম ctx->kv[চাবি_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[চাবি_id].get_type();
}

গণনা gguf_type gguf_get_arr_type(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].is_array);
    প্রত্যাগম ctx->kv[চাবি_id].get_type();
}

ধ্রুব শূন্য * gguf_get_arr_data(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_type() != GGUF_TYPE_STRING);
    প্রত্যাগম ctx->kv[চাবি_id].data.data();
}

ধ্রুব বর্ণ * gguf_get_arr_str(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id, size_t i) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_type() == GGUF_TYPE_STRING);
    প্রত্যাগম ctx->kv[চাবি_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));

    যদি (ctx->kv[চাবি_id].type == GGUF_TYPE_STRING) {
        প্রত্যাগম ctx->kv[চাবি_id].data_string.size();
    }

    ধ্রুব size_t type_size = gguf_type_size(ctx->kv[চাবি_id].type);
    GGML_ASSERT(ctx->kv[চাবি_id].data.size() % type_size == 0);
    প্রত্যাগম ctx->kv[চাবি_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<int32_t>();
}

ভিন্ন gguf_get_val_f32(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<ভিন্ন>();
}

uint64_t gguf_get_val_u64(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<int64_t>();
}

দ্বিগুণ gguf_get_val_f64(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<দ্বিগুণ>();
}

তার্কিক gguf_get_val_bool(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<তার্কিক>();
}

ধ্রুব বর্ণ * gguf_get_val_str(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    প্রত্যাগম ctx->kv[চাবি_id].get_val<std::string>().c_str();
}

ধ্রুব শূন্য * gguf_get_val_data(ধ্রুব সংরচনা gguf_context * ctx, int64_t চাবি_id) {
    GGML_ASSERT(চাবি_id >= 0 && চাবি_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[চাবি_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[চাবি_id].get_type() != GGUF_TYPE_STRING);
    প্রত্যাগম ctx->kv[চাবি_id].data.data();
}

int64_t gguf_get_n_tensors(ধ্রুব সংরচনা gguf_context * ctx) {
    প্রত্যাগম ctx->info.size();
}

int64_t gguf_find_tensor(ধ্রুব সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ধ্রুব int64_t n_tensors = gguf_get_n_tensors(ctx);

    প্রতি (int64_t i = 0; i < n_tensors; ++i) {
        যদি (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            ভঙ্গ;
        }
    }

    প্রত্যাগম tensor_id;
}

size_t gguf_get_tensor_offset(ধ্রুব সংরচনা gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    প্রত্যাগম ctx->info[tensor_id].offset;
}

ধ্রুব বর্ণ * gguf_get_tensor_name(ধ্রুব সংরচনা gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    প্রত্যাগম ctx->info[tensor_id].t.name;
}

ধ্রুব int64_t * gguf_get_tensor_ne(ধ্রুব সংরচনা gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    প্রত্যাগম ctx->info[tensor_id].t.ne;
}

গণনা ggml_type gguf_get_tensor_type(ধ্রুব সংরচনা gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    প্রত্যাগম ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(ধ্রুব সংরচনা gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    প্রত্যাগম ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_চাবি(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি) {
    ধ্রুব int64_t চাবি_id = gguf_find_চাবি(ctx, চাবি);
    যদি (চাবি_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + চাবি_id);
    }
    প্রত্যাগম চাবি_id;
}

template<typename T>
স্থির শূন্য gguf_check_reserved_চাবিগুলি(ধ্রুব std::string & চাবি, ধ্রুব T val) {
    যদি (চাবি == GGUF_চাবি_GENERAL_ALIGNMENT) {
        যদি constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_চাবি_GENERAL_ALIGNMENT " must be power of 2");
        } অন্যথা {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_চাবি_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

শূন্য gguf_set_val_u8(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, uint8_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i8(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, int8_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_u16(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, uint16_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i16(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, int16_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_u32(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, uint32_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i32(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, int32_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_f32(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, ভিন্ন val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_u64(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, uint64_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_i64(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, int64_t val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_f64(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, দ্বিগুণ val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_bool(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, তার্কিক val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, val);
}

শূন্য gguf_set_val_str(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, ধ্রুব বর্ণ * val) {
    gguf_check_reserved_চাবিগুলি(চাবি, val);
    gguf_remove_চাবি(ctx, চাবি);
    ctx->kv.emplace_back(চাবি, std::string(val));
}

শূন্য gguf_set_arr_data(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, গণনা gguf_type type, ধ্রুব শূন্য * data, size_t n) {
    gguf_check_reserved_চাবিগুলি(চাবি, data);
    gguf_remove_চাবি(ctx, চাবি);

    ধ্রুব size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    যদি (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(চাবি, tmp);
    ctx->kv.back().cast(type);
}

শূন্য gguf_set_arr_str(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * চাবি, ধ্রুব বর্ণ ** data, size_t n) {
    gguf_check_reserved_চাবিগুলি(চাবি, data);
    gguf_remove_চাবি(ctx, চাবি);

    std::vector<std::string> tmp(n);
    প্রতি (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(চাবি, tmp);
}

// set or add KV pairs from another context
শূন্য gguf_set_kv(সংরচনা gguf_context * ctx, ধ্রুব সংরচনা gguf_context * src) {
    ধ্রুব int64_t n_kv = gguf_get_n_kv(src);
    প্রতি (int64_t i = 0; i < n_kv; ++i) {
        ধ্রুব সংরচনা gguf_kv & kv = src->kv[i];

        যদি (!kv.is_array) {
            বিকল্প (kv.get_type()) {
                পক্ষ GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_চাবি().c_str(), kv.get_val<uint8_t>());             ভঙ্গ;
                পক্ষ GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_চাবি().c_str(), kv.get_val<int8_t>());              ভঙ্গ;
                পক্ষ GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_চাবি().c_str(), kv.get_val<uint16_t>());            ভঙ্গ;
                পক্ষ GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_চাবি().c_str(), kv.get_val<int16_t>());             ভঙ্গ;
                পক্ষ GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_চাবি().c_str(), kv.get_val<uint32_t>());            ভঙ্গ;
                পক্ষ GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_চাবি().c_str(), kv.get_val<int32_t>());             ভঙ্গ;
                পক্ষ GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_চাবি().c_str(), kv.get_val<ভিন্ন>());               ভঙ্গ;
                পক্ষ GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_চাবি().c_str(), kv.get_val<uint64_t>());            ভঙ্গ;
                পক্ষ GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_চাবি().c_str(), kv.get_val<int64_t>());             ভঙ্গ;
                পক্ষ GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_চাবি().c_str(), kv.get_val<দ্বিগুণ>());              ভঙ্গ;
                পক্ষ GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_চাবি().c_str(), kv.get_val<তার্কিক>());                ভঙ্গ;
                পক্ষ GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_চাবি().c_str(), kv.get_val<std::string>().c_str()); ভঙ্গ;
                পক্ষ GGUF_TYPE_ARRAY:
                স্বভাব: GGML_ABORT("invalid type");
            }
            অনুবর্ত;
        }

        ধ্রুব size_t ne = kv.get_ne();

        বিকল্প (kv.get_type()) {
            পক্ষ GGUF_TYPE_UINT8:
            পক্ষ GGUF_TYPE_INT8:
            পক্ষ GGUF_TYPE_UINT16:
            পক্ষ GGUF_TYPE_INT16:
            পক্ষ GGUF_TYPE_UINT32:
            পক্ষ GGUF_TYPE_INT32:
            পক্ষ GGUF_TYPE_FLOAT32:
            পক্ষ GGUF_TYPE_UINT64:
            পক্ষ GGUF_TYPE_INT64:
            পক্ষ GGUF_TYPE_FLOAT64:
            পক্ষ GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_চাবি().c_str(), kv.get_type(), kv.data.data(), ne);
            } ভঙ্গ;
            পক্ষ GGUF_TYPE_STRING: {
                std::vector<ধ্রুব বর্ণ *> tmp(ne);
                প্রতি (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_চাবি().c_str(), tmp.data(), ne);
            } ভঙ্গ;
            পক্ষ GGUF_TYPE_ARRAY:
            স্বভাব: GGML_ABORT("invalid type");
        }
    }
}

শূন্য gguf_add_tensor(
             সংরচনা gguf_context * ctx,
        ধ্রুব সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    যদি (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    সংরচনা gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

শূন্য gguf_set_tensor_type(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * name, গণনা ggml_type type) {
    ধ্রুব int64_t tensor_id = gguf_find_tensor(ctx, name);
    যদি (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    সংরচনা ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ধ্রুব size_t  type_size = ggml_type_size(type);
    ধ্রুব int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    প্রতি (পূর্ণাঙ্ক i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ধ্রুব int64_t n_tensors = gguf_get_n_tensors(ctx);
    প্রতি (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

শূন্য gguf_set_tensor_data(সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * name, ধ্রুব শূন্য * data) {
    ধ্রুব int64_t tensor_id = gguf_find_tensor(ctx, name);
    যদি (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (শূন্য *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

সংরচনা gguf_writer_base {
    size_t written_bytes {0u};

    আভাস ~gguf_writer_base(শূন্য) = স্বভাব;

    // we bet on devirtualization
    আভাস শূন্য লিখ(int8_t val) = 0;
    আভাস শূন্য লিখ(ধ্রুব std::vector<int8_t> & val) = 0;
    আভাস শূন্য write_tensor_data(ধ্রুব সংরচনা gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    শূন্য লিখ(ধ্রুব T & val) {
        প্রতি (size_t i = 0; i < পরিমাণ(val); ++i) {
            লিখ(reinterpret_cast<ধ্রুব int8_t *>(&val)[i]);
        }
    }

    শূন্য লিখ(ধ্রুব তার্কিক & val) {
        ধ্রুব int8_t val8 = val ? 1 : 0;
        লিখ(val8);
    }

    শূন্য লিখ(ধ্রুব std::string & val) {
        {
            ধ্রুব uint64_t n = val.length();
            লিখ(n);
        }
        প্রতি (size_t i = 0; i < val.length(); ++i) {
            লিখ((val.data())[i]);
        }
    }

    শূন্য লিখ(ধ্রুব বর্ণ * val) {
        লিখ(std::string(val));
    }

    শূন্য লিখ(ধ্রুব গণনা ggml_type & val) {
        লিখ(int32_t(val));
    }

    শূন্য লিখ(ধ্রুব গণনা gguf_type & val) {
        লিখ(int32_t(val));
    }

    শূন্য লিখ(ধ্রুব সংরচনা gguf_kv & kv) {
        ধ্রুব uint64_t ne = kv.get_ne();

        লিখ(kv.get_চাবি());

        যদি (kv.is_array) {
            লিখ(GGUF_TYPE_ARRAY);
            লিখ(kv.get_type());
            লিখ(ne);
        } অন্যথা {
            লিখ(kv.get_type());
        }

        বিকল্প (kv.get_type()) {
            পক্ষ GGUF_TYPE_UINT8:
            পক্ষ GGUF_TYPE_INT8:
            পক্ষ GGUF_TYPE_UINT16:
            পক্ষ GGUF_TYPE_INT16:
            পক্ষ GGUF_TYPE_UINT32:
            পক্ষ GGUF_TYPE_INT32:
            পক্ষ GGUF_TYPE_FLOAT32:
            পক্ষ GGUF_TYPE_UINT64:
            পক্ষ GGUF_TYPE_INT64:
            পক্ষ GGUF_TYPE_FLOAT64: {
                লিখ(kv.data);
            } ভঙ্গ;
            পক্ষ GGUF_TYPE_BOOL: {
                প্রতি (size_t i = 0; i < ne; ++i) {
                    লিখ(kv.get_val<তার্কিক>(i));
                }
            } ভঙ্গ;
            পক্ষ GGUF_TYPE_STRING: {
                প্রতি (size_t i = 0; i < ne; ++i) {
                    লিখ(kv.get_val<std::string>(i));
                }
            } ভঙ্গ;
            পক্ষ GGUF_TYPE_ARRAY:
            স্বভাব: GGML_ABORT("invalid type");
        }
    }

    শূন্য write_tensor_meta(ধ্রুব সংরচনা gguf_tensor_info & info) {
        লিখ(info.t.name);

        ধ্রুব uint32_t n_dims = ggml_n_dims(&info.t);
        লিখ(n_dims);

        প্রতি (uint32_t j = 0; j < n_dims; ++j) {
            লিখ(info.t.ne[j]);
        }
        লিখ(info.t.type);
        লিখ(info.offset);
    }

    শূন্য pad(ধ্রুব size_t alignment) {
        যাবত্ (written_bytes % alignment != 0) {
            ধ্রুব int8_t zero = 0;
            লিখ(zero);
        }
    }
};

// vector buffer based writer
সংরচনা gguf_writer_buf final : সার্বজনিক gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::লিখ;

    শূন্য লিখ(ধ্রুব int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    শূন্য লিখ(ধ্রুব std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    শূন্য write_tensor_data(ধ্রুব সংরচনা gguf_tensor_info & info, ধ্রুব size_t offset_data, ধ্রুব size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ধ্রুব size_t offset = buf.size();
        ধ্রুব size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        যদি (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } অন্যথা {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
সংরচনা gguf_writer_file final : সার্বজনিক gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::লিখ;

    শূন্য লিখ(ধ্রুব int8_t val) override {
        ধ্রুব auto real_val = static_cast<uint8_t>(val);
        ধ্রুব auto ret = fputc(real_val, file);
        written_bytes++;
        যদি (ret != real_val) {
            ক্ষিপ std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((পূর্ণাঙ্ক)real_val) + "'");
        }
    }

    শূন্য লিখ(ধ্রুব std::vector<int8_t> & val) override {
        ধ্রুব auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        যদি (ret != val.size()) {
            ক্ষিপ std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    শূন্য write_tensor_data(ধ্রুব সংরচনা gguf_tensor_info & info, ধ্রুব size_t offset_data, ধ্রুব size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ধ্রুব size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        যদি (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } অন্যথা {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        লিখ(buf);

        pad(alignment);
    }
};

template <typename writer_t>
স্থির শূন্য gguf_write_out(ধ্রুব সংরচনা gguf_context * ctx, writer_t & gw, তার্কিক only_meta) {
    ধ্রুব int64_t n_kv      = gguf_get_n_kv(ctx);
    ধ্রুব int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.লিখ(GGUF_MAGIC[0]);
    gw.লিখ(GGUF_MAGIC[1]);
    gw.লিখ(GGUF_MAGIC[2]);
    gw.লিখ(GGUF_MAGIC[3]);
    gw.লিখ(ctx->version);
    gw.লিখ(n_tensors);
    gw.লিখ(n_kv);

    // write key-value pairs
    প্রতি (int64_t i = 0; i < n_kv; ++i) {
        gw.লিখ(ctx->kv[i]);
    }

    // write tensor info
    প্রতি (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    যদি (only_meta) {
        প্রত্যাগম;
    }

    ধ্রুব size_t offset_data = gw.written_bytes;

    // write tensor data
    প্রতি (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

শূন্য gguf_write_to_buf(ধ্রুব সংরচনা gguf_context * ctx, std::vector<int8_t> & buf, তার্কিক only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

তার্কিক gguf_write_to_file_ptr(ধ্রুব সংরচনা gguf_context * ctx, FILE * file, তার্কিক only_meta) {
    GGML_ASSERT(file);

    যত্ন {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } গৃহাণ (ধ্রুব std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        প্রত্যাগম অসত্য;
    }
    প্রত্যাগম সত্য;
}

তার্কিক gguf_write_to_file(ধ্রুব সংরচনা gguf_context * ctx, ধ্রুব বর্ণ * fname, তার্কিক only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    যদি (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        প্রত্যাগম অসত্য;
    }

    ধ্রুব তার্কিক success = gguf_write_to_file_ptr(ctx, file, only_meta);
    যদি (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    প্রত্যাগম success;
}

size_t gguf_get_meta_size(ধ্রুব সংরচনা gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ সত্য);
    প্রত্যাগম buf.size();
}

শূন্য gguf_get_meta_data(ধ্রুব সংরচনা gguf_context * ctx, শূন্য * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ সত্য);
    memcpy(data, buf.data(), buf.size());
}
