#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "gguf.h"

#அந்தர்பாவ <cinttypes>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <cerrno>
#அந்தர்பாவ <map>
#அந்தர்பாவ <new>
#அந்தர்பாவ <stdexcept>
#அந்தர்பாவ <string>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#இல்லையெனில்
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
கட்டமைப்பு type_to_gguf_type;

template <>
கட்டமைப்பு type_to_gguf_type<uint8_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_UINT8;
};

template <>
கட்டமைப்பு type_to_gguf_type<int8_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_INT8;
};

template <>
கட்டமைப்பு type_to_gguf_type<uint16_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_UINT16;
};

template <>
கட்டமைப்பு type_to_gguf_type<int16_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_INT16;
};

template <>
கட்டமைப்பு type_to_gguf_type<uint32_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_UINT32;
};

template <>
கட்டமைப்பு type_to_gguf_type<int32_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_INT32;
};

template <>
கட்டமைப்பு type_to_gguf_type<தசம> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
கட்டமைப்பு type_to_gguf_type<தருக்க> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_BOOL;
};

template <>
கட்டமைப்பு type_to_gguf_type<std::string> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_STRING;
};

template <>
கட்டமைப்பு type_to_gguf_type<uint64_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_UINT64;
};

template <>
கட்டமைப்பு type_to_gguf_type<int64_t> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_INT64;
};

template <>
கட்டமைப்பு type_to_gguf_type<இரட்டை> {
    நிலையான constexpr எண்ணிக்கை gguf_type value = GGUF_TYPE_FLOAT64;
};

நிலையான மாறிலி std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   அளவு(uint8_t)},
    {GGUF_TYPE_INT8,    அளவு(int8_t)},
    {GGUF_TYPE_UINT16,  அளவு(uint16_t)},
    {GGUF_TYPE_INT16,   அளவு(int16_t)},
    {GGUF_TYPE_UINT32,  அளவு(uint32_t)},
    {GGUF_TYPE_INT32,   அளவு(int32_t)},
    {GGUF_TYPE_FLOAT32, அளவு(தசம)},
    {GGUF_TYPE_BOOL,    அளவு(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  அளவு(uint64_t)},
    {GGUF_TYPE_INT64,   அளவு(int64_t)},
    {GGUF_TYPE_FLOAT64, அளவு(இரட்டை)},
};
static_assert(GGUF_TYPE_மொத்தம் == 13, "GGUF_TYPE_COUNT != 13");

நிலையான மாறிலி std::map<gguf_type, மாறிலி எழுத்து *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_மொத்தம் == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(எண்ணிக்கை gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    திரும்பு it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

கட்டமைப்பு gguf_kv {
    std::string சாவி;

    தருக்க is_array;
    எண்ணிக்கை gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(மாறிலி std::string & சாவி, மாறிலி T value)
            : சாவி(சாவி), is_array(பொய்), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!சாவி.empty());
        data.resize(அளவு(T));
        memcpy(data.data(), &value, அளவு(T));
    }

    template <typename T>
    gguf_kv(மாறிலி std::string & சாவி, மாறிலி std::vector<T> & value)
            : சாவி(சாவி), is_array(உண்மை), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!சாவி.empty());
        data.resize(value.size()*அளவு(T));
        க்கு (size_t i = 0; i < value.size(); ++i) {
            மாறிலி T tmp = value[i];
            memcpy(data.data() + i*அளவு(T), &tmp, அளவு(T));
        }
    }

    gguf_kv(மாறிலி std::string & சாவி, மாறிலி std::string & value)
            : சாவி(சாவி), is_array(பொய்), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!சாவி.empty());
        data_string.push_back(value);
    }

    gguf_kv(மாறிலி std::string & சாவி, மாறிலி std::vector<std::string> & value)
            : சாவி(சாவி), is_array(உண்மை), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!சாவி.empty());
        data_string = value;
    }

    மாறிலி std::string & get_சாவி() மாறிலி {
        திரும்பு சாவி;
    }

    மாறிலி எண்ணிக்கை gguf_type & get_type() மாறிலி {
        திரும்பு type;
    }

    size_t get_ne() மாறிலி {
        எனில் (type == GGUF_TYPE_STRING) {
            மாறிலி size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            திரும்பு ne;
        }
        மாறிலி size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        மாறிலி size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        திரும்பு ne;
    }

    template <typename T>
    மாறிலி T & get_val(மாறிலி size_t i = 0) மாறிலி {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        எனில் constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            திரும்பு data_string[i];
        }
        மாறிலி size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        திரும்பு reinterpret_cast<மாறிலி T *>(data.data())[i];
    }

    வெறுமை cast(மாறிலி எண்ணிக்கை gguf_type new_type) {
        மாறிலி size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

கட்டமைப்பு gguf_tensor_info {
    கட்டமைப்பு ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

கட்டமைப்பு gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<கட்டமைப்பு gguf_kv> kv;
    std::vector<கட்டமைப்பு gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    வெறுமை * data = nullptr;
};

கட்டமைப்பு gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            வெறுமை * userdata,
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
    நிலையான uint64_t file_remain(FILE * file) {
        மாறிலி int64_t cur = gguf_ftell(file);
        எனில் (cur < 0) {
            திரும்பு 0;
        }
        எனில் (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            திரும்பு 0;
        }
        மாறிலி int64_t end = gguf_ftell(file);
        எனில் (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            திரும்பு 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        திரும்பு static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    தருக்க பட(T & dst) மாறிலி {
        மாறிலி size_t size = அளவு(dst);
        எனில் (size > nbytes_remain) {
            திரும்பு பொய்;
        }
        திரும்பு read_raw(&dst, size) == size;
    }

    template <typename T>
    தருக்க பட(std::vector<T> & dst, மாறிலி size_t n) மாறிலி {
        எனில் (n > GGUF_MAX_ARRAY_ELEMENTS) {
            திரும்பு பொய்;
        }
        எனில் constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            எனில் (n > SIZE_MAX / அளவு(uint64_t)) {
                திரும்பு பொய்;
            }
            எனில் (nbytes_remain < n * அளவு(uint64_t)) {
                திரும்பு பொய்;
            }
        } இல்லையெனில் {
            எனில் (n > SIZE_MAX / அளவு(T)) {
                திரும்பு பொய்;
            }
            எனில் (nbytes_remain < n * அளவு(T)) {
                திரும்பு பொய்;
            }
        }
        dst.resize(n);
        க்கு (size_t i = 0; i < dst.size(); ++i) {
            எனில் constexpr (std::is_same<T, தருக்க>::value) {
                தருக்க tmp;
                எனில் (!பட(tmp)) {
                    திரும்பு பொய்;
                }
                dst[i] = tmp;
            } இல்லையெனில் {
                எனில் (!பட(dst[i])) {
                    திரும்பு பொய்;
                }
            }
        }
        திரும்பு உண்மை;
    }

    தருக்க பட(தருக்க & dst) மாறிலி {
        int8_t tmp = -1;
        எனில் (!பட(tmp)) {
            திரும்பு பொய்;
        }
        dst = tmp != 0;
        திரும்பு உண்மை;
    }

    தருக்க பட(எண்ணிக்கை ggml_type & dst) மாறிலி {
        int32_t tmp = -1;
        எனில் (!பட(tmp)) {
            திரும்பு பொய்;
        }
        dst = ggml_type(tmp);
        திரும்பு உண்மை;
    }

    தருக்க பட(எண்ணிக்கை gguf_type & dst) மாறிலி {
        int32_t tmp = -1;
        எனில் (!பட(tmp)) {
            திரும்பு பொய்;
        }
        dst = gguf_type(tmp);
        திரும்பு உண்மை;
    }

    தருக்க பட(std::string & dst) மாறிலி {
        uint64_t size = 0;
        எனில் (!பட(size)) {
            திரும்பு பொய்;
        }
        எனில் (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            திரும்பு பொய்;
        }
        எனில் (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            திரும்பு பொய்;
        }
        dst.resize(static_cast<size_t>(size));
        திரும்பு read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    தருக்க பட(வெறுமை * dst, மாறிலி size_t size) மாறிலி {
        எனில் (size > nbytes_remain) {
            திரும்பு பொய்;
        }
        திரும்பு read_raw(dst, size) == size;
    }

    uint64_t tell() மாறிலி {
        திரும்பு data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() மாறிலி {
        திரும்பு start_offset;
    }

    தருக்க seek(uint64_t absolute_offset) மாறிலி {
        மாறிலி uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        எனில் (absolute_offset > end_offset) {
            திரும்பு பொய்;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        திரும்பு உண்மை;
    }

தனிப்பட்ட:
    size_t read_raw(வெறுமை * dst, size_t size) மாறிலி {
        எனில் (callback == nullptr || size == 0) {
            திரும்பு 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        தருக்க reached_eof = பொய்;

        போது (total_nread < size) {
            மாறிலி size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            எனில் (data_offset + total_nread < data_offset) {
                நிறுத்து;
            }
            மாறிலி size_t nread = callback(userdata, static_cast<வெறுமை *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            எனில் (nread != chunk_size) {
                reached_eof = உண்மை;
                நிறுத்து;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        எனில் (reached_eof) {
            nbytes_remain = 0;
        }

        திரும்பு total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    வெறுமை * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

கட்டமைப்பு gguf_context * gguf_init_empty(வெறுமை) {
    திரும்பு புதிய gguf_context;
}

template<typename T>
தருக்க gguf_read_emplace_helper(மாறிலி கட்டமைப்பு gguf_reader & gr, std::vector<கட்டமைப்பு gguf_kv> & kv, மாறிலி std::string & சாவி, மாறிலி தருக்க is_array, மாறிலி size_t n) {
    எனில் (is_array) {
        std::vector<T> value;
        முயல் {
            எனில் (!gr.பட(value, n)) {
                திரும்பு பொய்;
            }
        } பிடி (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, சாவி.c_str());
            திரும்பு பொய்;
        } பிடி (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, சாவி.c_str());
            திரும்பு பொய்;
        }
        kv.emplace_back(சாவி, value);
    } இல்லையெனில் {
        T value;
        எனில் (!gr.பட(value)) {
            திரும்பு பொய்;
        }
        kv.emplace_back(சாவி, value);
    }
    திரும்பு உண்மை;
}

நிலையான கட்டமைப்பு gguf_context * gguf_init_from_reader(மாறிலி கட்டமைப்பு gguf_reader & gr, கட்டமைப்பு gguf_init_params params) {
    கட்டமைப்பு gguf_context * ctx = புதிய gguf_context;

    தருக்க ok = உண்மை;

    // file magic
    {
        std::vector<எழுத்து> magic;
        ok = ok && gr.பட(magic, 4);

        எனில் (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            திரும்பு nullptr;
        }

        க்கு (uint32_t i = 0; i < magic.size(); i++) {
            எனில் (magic[i] != GGUF_MAGIC[i]) {
                எழுத்து c0 = isprint(magic[0]) ? magic[0] : '?';
                எழுத்து c1 = isprint(magic[1]) ? magic[1] : '?';
                எழுத்து c2 = isprint(magic[2]) ? magic[2] : '?';
                எழுத்து c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                திரும்பு nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    எனில் (ok && gr.பட(ctx->version)) {
        எனில் (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = பொய்;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        எனில் (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = பொய்;
        }

        எனில் (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = பொய்;
        }
        எனில் (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = பொய்;
        }
    } இல்லையெனில் {
        ok = பொய்;
    }

    எனில் (ok && gr.பட(n_tensors)) {
        static_assert(அளவு(size_t) <= 8 && அளவு(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        எனில் (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/அளவு(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/அளவு(gguf_tensor_info));
            ok = பொய்;
        }
    } இல்லையெனில் {
        ok = பொய்;
    }

    எனில் (ok && gr.பட(n_kv)) {
        static_assert(அளவு(size_t) <= 8 && அளவு(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        எனில் (n_kv < 0 || n_kv > int64_t(SIZE_MAX/அளவு(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/அளவு(gguf_kv));
            ok = பொய்;
        }
    } இல்லையெனில் {
        ok = பொய்;
    }

    எனில் (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        திரும்பு nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_சாவிகள்;

        க்கு (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string சாவி;
            gguf_type   type     = gguf_type(-1);
            தருக்க        is_array = பொய்;
            uint64_t    n        = 1;

            முயல் {
                ok = ok && gr.பட(சாவி);
            } பிடி (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = பொய்;
            } பிடி (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = பொய்;
            }
            எனில் (ok && சாவி.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = பொய்;
            }
            எனில் (ok && !seen_சாவிகள்.insert(சாவி).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, சாவி.c_str(), i);
                ok = பொய்;
            }
            எனில் (!ok) {
                நிறுத்து;
            }

            ok = ok && gr.பட(type);
            எனில் (type == GGUF_TYPE_ARRAY) {
                is_array = உண்மை;
                ok = ok && gr.பட(type);
                ok = ok && gr.பட(n);
            }
            எனில் (!ok) {
                நிறுத்து;
            }

            தேர்வு (type) {
                நிலை GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<தசம>      (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<தருக்க>       (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<இரட்டை>     (gr, ctx->kv, சாவி, is_array, n); நிறுத்து;
                நிலை GGUF_TYPE_ARRAY:
                இயல்பு:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, சாவி.c_str(), type);
                        ok = பொய்;
                    } நிறுத்து;
            }
        }

        எனில் (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            திரும்பு nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        மாறிலி முழு alignment_idx = gguf_find_சாவி(ctx, GGUF_சாவி_GENERAL_ALIGNMENT);
        எனில் (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_சாவி_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            திரும்பு nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        எனில் (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            திரும்பு nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    க்கு (int64_t i = 0; ok && i < n_tensors; ++i) {
        கட்டமைப்பு gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            முயல் {
                ok = ok && gr.பட(name);
            } பிடி (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = பொய்;
            } பிடி (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = பொய்;
            }
            எனில் (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = பொய்;
                நிறுத்து;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            எனில் (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = பொய்;
                நிறுத்து;
            }
        }
        எனில் (!ok) {
            நிறுத்து;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.பட(n_dims);
            எனில் (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = பொய்;
                நிறுத்து;
            }
            க்கு (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                எனில் (j < n_dims) {
                    ok = ok && gr.பட(info.t.ne[j]);
                }

                // check that all ne are non-negative
                எனில் (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = பொய்;
                    நிறுத்து;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            எனில் (ok) {
                int64_t current_nelements = info.t.ne[0];
                க்கு (முழு i = 1; i < GGML_MAX_DIMS; ++i) {
                    எனில் (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        எனில் (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = பொய்;
                             நிறுத்து;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 எனில் (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     நிறுத்து;
                 }
            }
        }
        எனில் (!ok) {
            நிறுத்து;
        }

        // tensor type
        {
            ok = ok && gr.பட(info.t.type);

            // check that tensor type is within defined range
            எனில் (info.t.type < 0 || info.t.type >= GGML_TYPE_மொத்தம்) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_மொத்தம்);
                ok = பொய்;
                நிறுத்து;
            }
            மாறிலி size_t  type_size = ggml_type_size(info.t.type);
            மாறிலி int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            எனில் (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (முழு) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = பொய்;
                நிறுத்து;
            }

            // check that the size of the tensor in bytes is representable
            எனில் (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = பொய்;
                நிறுத்து;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            க்கு (முழு j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        எனில் (!ok) {
            நிறுத்து;
        }

        // tensor data offset within buffer
        ok = ok && gr.பட(info.offset);

        ctx->info.push_back(info);
    }

    எனில் (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        திரும்பு nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    எனில் (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        திரும்பு nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        க்கு (size_t i = 0; i < ctx->info.size(); ++i) {
            மாறிலி gguf_tensor_info & ti = ctx->info[i];
            எனில் (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                திரும்பு nullptr;
            }
            மாறிலி size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            எனில் (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                திரும்பு nullptr;
            }
            மாறிலி size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            எனில் (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                திரும்பு nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    எனில் (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        எனில் (params.no_alloc) {
            எனில் (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                திரும்பு nullptr;
            }

            மாறிலி size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } இல்லையெனில் {
            எனில் ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                திரும்பு nullptr;
            }

            மாறிலி size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            எனில் (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                திரும்பு nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        கட்டமைப்பு ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        எனில் (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            திரும்பு nullptr;
        }

        கட்டமைப்பு ggml_context * ctx_data = *params.ctx;

        கட்டமைப்பு ggml_tensor * data = nullptr;

        எனில் (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            எனில் (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.பட(data->data, ctx->size);

            எனில் (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                திரும்பு nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, உண்மை);

        // create the tensors
        க்கு (size_t i = 0; i < ctx->info.size(); ++i) {
            மாறிலி கட்டமைப்பு gguf_tensor_info & info = ctx->info[i];

            கட்டமைப்பு ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            எனில் (!ok) {
                நிறுத்து;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            எனில் (!params.no_alloc) {
                cur->data = (எழுத்து *) data->data + info.offset;
            }
        }

        எனில் (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            திரும்பு nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    திரும்பு ctx;
}

கட்டமைப்பு gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, வெறுமை * userdata, size_t max_chunk_read, uint64_t max_expected_size, கட்டமைப்பு gguf_init_params params) {
    எனில் (callback == nullptr) {
        திரும்பு nullptr;
    }

    மாறிலி கட்டமைப்பு gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    திரும்பு gguf_init_from_reader(gr, params);
}

கட்டமைப்பு gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

நிலையான size_t gguf_file_reader_callback(வெறுமை * userdata, வெறுமை * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    எனில் (reader.offset != offset) {
        எனில் (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            திரும்பு 0;
        }

        reader.offset = offset;
    }

    மாறிலி size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    திரும்பு nread;
}

கட்டமைப்பு gguf_context * gguf_init_from_file_ptr(FILE * file, கட்டமைப்பு gguf_init_params params) {
    எனில் (!file) {
        திரும்பு nullptr;
    }

    மாறிலி int64_t cur = gguf_ftell(file);
    எனில் (cur < 0) {
        திரும்பு nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    மாறிலி கட்டமைப்பு gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    திரும்பு gguf_init_from_reader(gr, params);
}

கட்டமைப்பு gguf_buffer_reader {
    மாறிலி uint8_t * data;
    size_t          size;
};

நிலையான size_t gguf_buffer_reader_callback(வெறுமை * userdata, வெறுமை * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    மாறிலி gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    எனில் (offset > reader.size || len > reader.size - offset) {
        திரும்பு 0;
    }

    மாறிலி size_t data_offset = static_cast<size_t>(offset);
    மாறிலி size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    திரும்பு nread;
}

கட்டமைப்பு gguf_context * gguf_init_from_buffer(மாறிலி வெறுமை * data, size_t size, கட்டமைப்பு gguf_init_params params) {
    எனில் (data == nullptr || size == 0) {
        திரும்பு nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<மாறிலி uint8_t *>(data),
        /*.size = */ size,
    };
    மாறிலி கட்டமைப்பு gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    திரும்பு gguf_init_from_reader(gr, params);
}

கட்டமைப்பு gguf_context * gguf_init_from_file(மாறிலி எழுத்து * fname, கட்டமைப்பு gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    எனில் (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        திரும்பு nullptr;
    }

    கட்டமைப்பு gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    திரும்பு result;
}

வெறுமை gguf_free(கட்டமைப்பு gguf_context * ctx) {
    எனில் (ctx == nullptr) {
        திரும்பு;
    }
    delete ctx;
}

மாறிலி எழுத்து * gguf_type_name(எண்ணிக்கை gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    திரும்பு it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(மாறிலி கட்டமைப்பு gguf_context * ctx) {
    திரும்பு ctx->version;
}

size_t gguf_get_alignment(மாறிலி கட்டமைப்பு gguf_context * ctx) {
    திரும்பு ctx->alignment;
}

size_t gguf_get_data_offset(மாறிலி கட்டமைப்பு gguf_context * ctx) {
    திரும்பு ctx->offset;
}

int64_t gguf_get_n_kv(மாறிலி கட்டமைப்பு gguf_context * ctx) {
    திரும்பு ctx->kv.size();
}

int64_t gguf_find_சாவி(மாறிலி கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி) {
    // return -1 if key not found
    int64_t keyfound = -1;

    மாறிலி int64_t n_kv = gguf_get_n_kv(ctx);

    க்கு (int64_t i = 0; i < n_kv; ++i) {
        எனில் (strcmp(சாவி, gguf_get_சாவி(ctx, i)) == 0) {
            keyfound = i;
            நிறுத்து;
        }
    }

    திரும்பு keyfound;
}

மாறிலி எழுத்து * gguf_get_சாவி(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    திரும்பு ctx->kv[சாவி_id].get_சாவி().c_str();
}

எண்ணிக்கை gguf_type gguf_get_kv_type(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    திரும்பு ctx->kv[சாவி_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[சாவி_id].get_type();
}

எண்ணிக்கை gguf_type gguf_get_arr_type(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].is_array);
    திரும்பு ctx->kv[சாவி_id].get_type();
}

மாறிலி வெறுமை * gguf_get_arr_data(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_type() != GGUF_TYPE_STRING);
    திரும்பு ctx->kv[சாவி_id].data.data();
}

மாறிலி எழுத்து * gguf_get_arr_str(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id, size_t i) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_type() == GGUF_TYPE_STRING);
    திரும்பு ctx->kv[சாவி_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));

    எனில் (ctx->kv[சாவி_id].type == GGUF_TYPE_STRING) {
        திரும்பு ctx->kv[சாவி_id].data_string.size();
    }

    மாறிலி size_t type_size = gguf_type_size(ctx->kv[சாவி_id].type);
    GGML_ASSERT(ctx->kv[சாவி_id].data.size() % type_size == 0);
    திரும்பு ctx->kv[சாவி_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<int32_t>();
}

தசம gguf_get_val_f32(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<தசம>();
}

uint64_t gguf_get_val_u64(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<int64_t>();
}

இரட்டை gguf_get_val_f64(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<இரட்டை>();
}

தருக்க gguf_get_val_bool(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<தருக்க>();
}

மாறிலி எழுத்து * gguf_get_val_str(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    திரும்பு ctx->kv[சாவி_id].get_val<std::string>().c_str();
}

மாறிலி வெறுமை * gguf_get_val_data(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t சாவி_id) {
    GGML_ASSERT(சாவி_id >= 0 && சாவி_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[சாவி_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[சாவி_id].get_type() != GGUF_TYPE_STRING);
    திரும்பு ctx->kv[சாவி_id].data.data();
}

int64_t gguf_get_n_tensors(மாறிலி கட்டமைப்பு gguf_context * ctx) {
    திரும்பு ctx->info.size();
}

int64_t gguf_find_tensor(மாறிலி கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    மாறிலி int64_t n_tensors = gguf_get_n_tensors(ctx);

    க்கு (int64_t i = 0; i < n_tensors; ++i) {
        எனில் (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            நிறுத்து;
        }
    }

    திரும்பு tensor_id;
}

size_t gguf_get_tensor_offset(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    திரும்பு ctx->info[tensor_id].offset;
}

மாறிலி எழுத்து * gguf_get_tensor_name(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    திரும்பு ctx->info[tensor_id].t.name;
}

மாறிலி int64_t * gguf_get_tensor_ne(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    திரும்பு ctx->info[tensor_id].t.ne;
}

எண்ணிக்கை ggml_type gguf_get_tensor_type(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    திரும்பு ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(மாறிலி கட்டமைப்பு gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    திரும்பு ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_சாவி(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி) {
    மாறிலி int64_t சாவி_id = gguf_find_சாவி(ctx, சாவி);
    எனில் (சாவி_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + சாவி_id);
    }
    திரும்பு சாவி_id;
}

template<typename T>
நிலையான வெறுமை gguf_check_reserved_சாவிகள்(மாறிலி std::string & சாவி, மாறிலி T val) {
    எனில் (சாவி == GGUF_சாவி_GENERAL_ALIGNMENT) {
        எனில் constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_சாவி_GENERAL_ALIGNMENT " must be power of 2");
        } இல்லையெனில் {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_சாவி_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

வெறுமை gguf_set_val_u8(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, uint8_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_i8(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, int8_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_u16(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, uint16_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_i16(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, int16_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_u32(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, uint32_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_i32(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, int32_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_f32(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, தசம val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_u64(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, uint64_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_i64(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, int64_t val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_f64(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, இரட்டை val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_bool(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, தருக்க val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, val);
}

வெறுமை gguf_set_val_str(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, மாறிலி எழுத்து * val) {
    gguf_check_reserved_சாவிகள்(சாவி, val);
    gguf_remove_சாவி(ctx, சாவி);
    ctx->kv.emplace_back(சாவி, std::string(val));
}

வெறுமை gguf_set_arr_data(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, எண்ணிக்கை gguf_type type, மாறிலி வெறுமை * data, size_t n) {
    gguf_check_reserved_சாவிகள்(சாவி, data);
    gguf_remove_சாவி(ctx, சாவி);

    மாறிலி size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    எனில் (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(சாவி, tmp);
    ctx->kv.back().cast(type);
}

வெறுமை gguf_set_arr_str(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * சாவி, மாறிலி எழுத்து ** data, size_t n) {
    gguf_check_reserved_சாவிகள்(சாவி, data);
    gguf_remove_சாவி(ctx, சாவி);

    std::vector<std::string> tmp(n);
    க்கு (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(சாவி, tmp);
}

// set or add KV pairs from another context
வெறுமை gguf_set_kv(கட்டமைப்பு gguf_context * ctx, மாறிலி கட்டமைப்பு gguf_context * src) {
    மாறிலி int64_t n_kv = gguf_get_n_kv(src);
    க்கு (int64_t i = 0; i < n_kv; ++i) {
        மாறிலி கட்டமைப்பு gguf_kv & kv = src->kv[i];

        எனில் (!kv.is_array) {
            தேர்வு (kv.get_type()) {
                நிலை GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_சாவி().c_str(), kv.get_val<uint8_t>());             நிறுத்து;
                நிலை GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_சாவி().c_str(), kv.get_val<int8_t>());              நிறுத்து;
                நிலை GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_சாவி().c_str(), kv.get_val<uint16_t>());            நிறுத்து;
                நிலை GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_சாவி().c_str(), kv.get_val<int16_t>());             நிறுத்து;
                நிலை GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_சாவி().c_str(), kv.get_val<uint32_t>());            நிறுத்து;
                நிலை GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_சாவி().c_str(), kv.get_val<int32_t>());             நிறுத்து;
                நிலை GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_சாவி().c_str(), kv.get_val<தசம>());               நிறுத்து;
                நிலை GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_சாவி().c_str(), kv.get_val<uint64_t>());            நிறுத்து;
                நிலை GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_சாவி().c_str(), kv.get_val<int64_t>());             நிறுத்து;
                நிலை GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_சாவி().c_str(), kv.get_val<இரட்டை>());              நிறுத்து;
                நிலை GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_சாவி().c_str(), kv.get_val<தருக்க>());                நிறுத்து;
                நிலை GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_சாவி().c_str(), kv.get_val<std::string>().c_str()); நிறுத்து;
                நிலை GGUF_TYPE_ARRAY:
                இயல்பு: GGML_ABORT("invalid type");
            }
            தொடர்;
        }

        மாறிலி size_t ne = kv.get_ne();

        தேர்வு (kv.get_type()) {
            நிலை GGUF_TYPE_UINT8:
            நிலை GGUF_TYPE_INT8:
            நிலை GGUF_TYPE_UINT16:
            நிலை GGUF_TYPE_INT16:
            நிலை GGUF_TYPE_UINT32:
            நிலை GGUF_TYPE_INT32:
            நிலை GGUF_TYPE_FLOAT32:
            நிலை GGUF_TYPE_UINT64:
            நிலை GGUF_TYPE_INT64:
            நிலை GGUF_TYPE_FLOAT64:
            நிலை GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_சாவி().c_str(), kv.get_type(), kv.data.data(), ne);
            } நிறுத்து;
            நிலை GGUF_TYPE_STRING: {
                std::vector<மாறிலி எழுத்து *> tmp(ne);
                க்கு (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_சாவி().c_str(), tmp.data(), ne);
            } நிறுத்து;
            நிலை GGUF_TYPE_ARRAY:
            இயல்பு: GGML_ABORT("invalid type");
        }
    }
}

வெறுமை gguf_add_tensor(
             கட்டமைப்பு gguf_context * ctx,
        மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    எனில் (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    கட்டமைப்பு gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

வெறுமை gguf_set_tensor_type(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * name, எண்ணிக்கை ggml_type type) {
    மாறிலி int64_t tensor_id = gguf_find_tensor(ctx, name);
    எனில் (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    கட்டமைப்பு ggml_tensor * tensor = &ctx->info[tensor_id].t;
    மாறிலி size_t  type_size = ggml_type_size(type);
    மாறிலி int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    க்கு (முழு i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    மாறிலி int64_t n_tensors = gguf_get_n_tensors(ctx);
    க்கு (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

வெறுமை gguf_set_tensor_data(கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * name, மாறிலி வெறுமை * data) {
    மாறிலி int64_t tensor_id = gguf_find_tensor(ctx, name);
    எனில் (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (வெறுமை *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

கட்டமைப்பு gguf_writer_base {
    size_t written_bytes {0u};

    மெய்நிகர் ~gguf_writer_base(வெறுமை) = இயல்பு;

    // we bet on devirtualization
    மெய்நிகர் வெறுமை லிக(int8_t val) = 0;
    மெய்நிகர் வெறுமை லிக(மாறிலி std::vector<int8_t> & val) = 0;
    மெய்நிகர் வெறுமை write_tensor_data(மாறிலி கட்டமைப்பு gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    வெறுமை லிக(மாறிலி T & val) {
        க்கு (size_t i = 0; i < அளவு(val); ++i) {
            லிக(reinterpret_cast<மாறிலி int8_t *>(&val)[i]);
        }
    }

    வெறுமை லிக(மாறிலி தருக்க & val) {
        மாறிலி int8_t val8 = val ? 1 : 0;
        லிக(val8);
    }

    வெறுமை லிக(மாறிலி std::string & val) {
        {
            மாறிலி uint64_t n = val.length();
            லிக(n);
        }
        க்கு (size_t i = 0; i < val.length(); ++i) {
            லிக((val.data())[i]);
        }
    }

    வெறுமை லிக(மாறிலி எழுத்து * val) {
        லிக(std::string(val));
    }

    வெறுமை லிக(மாறிலி எண்ணிக்கை ggml_type & val) {
        லிக(int32_t(val));
    }

    வெறுமை லிக(மாறிலி எண்ணிக்கை gguf_type & val) {
        லிக(int32_t(val));
    }

    வெறுமை லிக(மாறிலி கட்டமைப்பு gguf_kv & kv) {
        மாறிலி uint64_t ne = kv.get_ne();

        லிக(kv.get_சாவி());

        எனில் (kv.is_array) {
            லிக(GGUF_TYPE_ARRAY);
            லிக(kv.get_type());
            லிக(ne);
        } இல்லையெனில் {
            லிக(kv.get_type());
        }

        தேர்வு (kv.get_type()) {
            நிலை GGUF_TYPE_UINT8:
            நிலை GGUF_TYPE_INT8:
            நிலை GGUF_TYPE_UINT16:
            நிலை GGUF_TYPE_INT16:
            நிலை GGUF_TYPE_UINT32:
            நிலை GGUF_TYPE_INT32:
            நிலை GGUF_TYPE_FLOAT32:
            நிலை GGUF_TYPE_UINT64:
            நிலை GGUF_TYPE_INT64:
            நிலை GGUF_TYPE_FLOAT64: {
                லிக(kv.data);
            } நிறுத்து;
            நிலை GGUF_TYPE_BOOL: {
                க்கு (size_t i = 0; i < ne; ++i) {
                    லிக(kv.get_val<தருக்க>(i));
                }
            } நிறுத்து;
            நிலை GGUF_TYPE_STRING: {
                க்கு (size_t i = 0; i < ne; ++i) {
                    லிக(kv.get_val<std::string>(i));
                }
            } நிறுத்து;
            நிலை GGUF_TYPE_ARRAY:
            இயல்பு: GGML_ABORT("invalid type");
        }
    }

    வெறுமை write_tensor_meta(மாறிலி கட்டமைப்பு gguf_tensor_info & info) {
        லிக(info.t.name);

        மாறிலி uint32_t n_dims = ggml_n_dims(&info.t);
        லிக(n_dims);

        க்கு (uint32_t j = 0; j < n_dims; ++j) {
            லிக(info.t.ne[j]);
        }
        லிக(info.t.type);
        லிக(info.offset);
    }

    வெறுமை pad(மாறிலி size_t alignment) {
        போது (written_bytes % alignment != 0) {
            மாறிலி int8_t zero = 0;
            லிக(zero);
        }
    }
};

// vector buffer based writer
கட்டமைப்பு gguf_writer_buf final : பொது gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::லிக;

    வெறுமை லிக(மாறிலி int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    வெறுமை லிக(மாறிலி std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    வெறுமை write_tensor_data(மாறிலி கட்டமைப்பு gguf_tensor_info & info, மாறிலி size_t offset_data, மாறிலி size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        மாறிலி size_t offset = buf.size();
        மாறிலி size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        எனில் (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } இல்லையெனில் {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
கட்டமைப்பு gguf_writer_file final : பொது gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::லிக;

    வெறுமை லிக(மாறிலி int8_t val) override {
        மாறிலி auto real_val = static_cast<uint8_t>(val);
        மாறிலி auto ret = fputc(real_val, file);
        written_bytes++;
        எனில் (ret != real_val) {
            எறி std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((முழு)real_val) + "'");
        }
    }

    வெறுமை லிக(மாறிலி std::vector<int8_t> & val) override {
        மாறிலி auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        எனில் (ret != val.size()) {
            எறி std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    வெறுமை write_tensor_data(மாறிலி கட்டமைப்பு gguf_tensor_info & info, மாறிலி size_t offset_data, மாறிலி size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        மாறிலி size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        எனில் (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } இல்லையெனில் {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        லிக(buf);

        pad(alignment);
    }
};

template <typename writer_t>
நிலையான வெறுமை gguf_write_out(மாறிலி கட்டமைப்பு gguf_context * ctx, writer_t & gw, தருக்க only_meta) {
    மாறிலி int64_t n_kv      = gguf_get_n_kv(ctx);
    மாறிலி int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.லிக(GGUF_MAGIC[0]);
    gw.லிக(GGUF_MAGIC[1]);
    gw.லிக(GGUF_MAGIC[2]);
    gw.லிக(GGUF_MAGIC[3]);
    gw.லிக(ctx->version);
    gw.லிக(n_tensors);
    gw.லிக(n_kv);

    // write key-value pairs
    க்கு (int64_t i = 0; i < n_kv; ++i) {
        gw.லிக(ctx->kv[i]);
    }

    // write tensor info
    க்கு (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    எனில் (only_meta) {
        திரும்பு;
    }

    மாறிலி size_t offset_data = gw.written_bytes;

    // write tensor data
    க்கு (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

வெறுமை gguf_write_to_buf(மாறிலி கட்டமைப்பு gguf_context * ctx, std::vector<int8_t> & buf, தருக்க only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

தருக்க gguf_write_to_file_ptr(மாறிலி கட்டமைப்பு gguf_context * ctx, FILE * file, தருக்க only_meta) {
    GGML_ASSERT(file);

    முயல் {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } பிடி (மாறிலி std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

தருக்க gguf_write_to_file(மாறிலி கட்டமைப்பு gguf_context * ctx, மாறிலி எழுத்து * fname, தருக்க only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    எனில் (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        திரும்பு பொய்;
    }

    மாறிலி தருக்க success = gguf_write_to_file_ptr(ctx, file, only_meta);
    எனில் (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    திரும்பு success;
}

size_t gguf_get_meta_size(மாறிலி கட்டமைப்பு gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ உண்மை);
    திரும்பு buf.size();
}

வெறுமை gguf_get_meta_data(மாறிலி கட்டமைப்பு gguf_context * ctx, வெறுமை * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ உண்மை);
    memcpy(data, buf.data(), buf.size());
}
