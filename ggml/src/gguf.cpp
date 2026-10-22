#ضمن "ggml.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "gguf.h"

#ضمن <cinttypes>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <cerrno>
#ضمن <map>
#ضمن <new>
#ضمن <stdexcept>
#ضمن <string>
#ضمن <unordered_set>
#ضمن <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#وإلا
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
بنية type_to_gguf_type;

template <>
بنية type_to_gguf_type<uint8_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT8;
};

template <>
بنية type_to_gguf_type<int8_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT8;
};

template <>
بنية type_to_gguf_type<uint16_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT16;
};

template <>
بنية type_to_gguf_type<int16_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT16;
};

template <>
بنية type_to_gguf_type<uint32_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT32;
};

template <>
بنية type_to_gguf_type<int32_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT32;
};

template <>
بنية type_to_gguf_type<عائم> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
بنية type_to_gguf_type<منطقي> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_BOOL;
};

template <>
بنية type_to_gguf_type<std::string> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_STRING;
};

template <>
بنية type_to_gguf_type<uint64_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT64;
};

template <>
بنية type_to_gguf_type<int64_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT64;
};

template <>
بنية type_to_gguf_type<مزدوج> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_FLOAT64;
};

ساكن ثابت std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   حجم(uint8_t)},
    {GGUF_TYPE_INT8,    حجم(int8_t)},
    {GGUF_TYPE_UINT16,  حجم(uint16_t)},
    {GGUF_TYPE_INT16,   حجم(int16_t)},
    {GGUF_TYPE_UINT32,  حجم(uint32_t)},
    {GGUF_TYPE_INT32,   حجم(int32_t)},
    {GGUF_TYPE_FLOAT32, حجم(عائم)},
    {GGUF_TYPE_BOOL,    حجم(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  حجم(uint64_t)},
    {GGUF_TYPE_INT64,   حجم(int64_t)},
    {GGUF_TYPE_FLOAT64, حجم(مزدوج)},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

ساكن ثابت std::map<gguf_type, ثابت حرف *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(تعداد gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    أرجع it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

بنية gguf_kv {
    std::string key;

    منطقي is_array;
    تعداد gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(ثابت std::string & key, ثابت T value)
            : key(key), is_array(خطأ), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(حجم(T));
        memcpy(data.data(), &value, حجم(T));
    }

    template <typename T>
    gguf_kv(ثابت std::string & key, ثابت std::vector<T> & value)
            : key(key), is_array(𐡒𐡔𐡈), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(value.size()*حجم(T));
        لكل (size_t i = 0; i < value.size(); ++i) {
            ثابت T tmp = value[i];
            memcpy(data.data() + i*حجم(T), &tmp, حجم(T));
        }
    }

    gguf_kv(ثابت std::string & key, ثابت std::string & value)
            : key(key), is_array(خطأ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string.push_back(value);
    }

    gguf_kv(ثابت std::string & key, ثابت std::vector<std::string> & value)
            : key(key), is_array(𐡒𐡔𐡈), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string = value;
    }

    ثابت std::string & get_key() ثابت {
        أرجع key;
    }

    ثابت تعداد gguf_type & get_type() ثابت {
        أرجع type;
    }

    size_t get_ne() ثابت {
        𐡄𐡍 (type == GGUF_TYPE_STRING) {
            ثابت size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            أرجع ne;
        }
        ثابت size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        ثابت size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        أرجع ne;
    }

    template <typename T>
    ثابت T & get_val(ثابت size_t i = 0) ثابت {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        𐡄𐡍 constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            أرجع data_string[i];
        }
        ثابت size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        أرجع reinterpret_cast<ثابت T *>(data.data())[i];
    }

    فراغ cast(ثابت تعداد gguf_type new_type) {
        ثابت size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

بنية gguf_tensor_info {
    بنية ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

بنية gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<بنية gguf_kv> kv;
    std::vector<بنية gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    فراغ * data = nullptr;
};

بنية gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            فراغ * userdata,
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
    ساكن uint64_t file_remain(FILE * file) {
        ثابت int64_t cur = gguf_ftell(file);
        𐡄𐡍 (cur < 0) {
            أرجع 0;
        }
        𐡄𐡍 (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            أرجع 0;
        }
        ثابت int64_t end = gguf_ftell(file);
        𐡄𐡍 (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            أرجع 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        أرجع static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    منطقي اقرأ(T & dst) ثابت {
        ثابت size_t size = حجم(dst);
        𐡄𐡍 (size > nbytes_remain) {
            أرجع خطأ;
        }
        أرجع read_raw(&dst, size) == size;
    }

    template <typename T>
    منطقي اقرأ(std::vector<T> & dst, ثابت size_t n) ثابت {
        𐡄𐡍 (n > GGUF_MAX_ARRAY_ELEMENTS) {
            أرجع خطأ;
        }
        𐡄𐡍 constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            𐡄𐡍 (n > SIZE_MAX / حجم(uint64_t)) {
                أرجع خطأ;
            }
            𐡄𐡍 (nbytes_remain < n * حجم(uint64_t)) {
                أرجع خطأ;
            }
        } وإلا {
            𐡄𐡍 (n > SIZE_MAX / حجم(T)) {
                أرجع خطأ;
            }
            𐡄𐡍 (nbytes_remain < n * حجم(T)) {
                أرجع خطأ;
            }
        }
        dst.resize(n);
        لكل (size_t i = 0; i < dst.size(); ++i) {
            𐡄𐡍 constexpr (std::is_same<T, منطقي>::value) {
                منطقي tmp;
                𐡄𐡍 (!اقرأ(tmp)) {
                    أرجع خطأ;
                }
                dst[i] = tmp;
            } وإلا {
                𐡄𐡍 (!اقرأ(dst[i])) {
                    أرجع خطأ;
                }
            }
        }
        أرجع 𐡒𐡔𐡈;
    }

    منطقي اقرأ(منطقي & dst) ثابت {
        int8_t tmp = -1;
        𐡄𐡍 (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = tmp != 0;
        أرجع 𐡒𐡔𐡈;
    }

    منطقي اقرأ(تعداد ggml_type & dst) ثابت {
        int32_t tmp = -1;
        𐡄𐡍 (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = ggml_type(tmp);
        أرجع 𐡒𐡔𐡈;
    }

    منطقي اقرأ(تعداد gguf_type & dst) ثابت {
        int32_t tmp = -1;
        𐡄𐡍 (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = gguf_type(tmp);
        أرجع 𐡒𐡔𐡈;
    }

    منطقي اقرأ(std::string & dst) ثابت {
        uint64_t size = 0;
        𐡄𐡍 (!اقرأ(size)) {
            أرجع خطأ;
        }
        𐡄𐡍 (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            أرجع خطأ;
        }
        𐡄𐡍 (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            أرجع خطأ;
        }
        dst.resize(static_cast<size_t>(size));
        أرجع read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    منطقي اقرأ(فراغ * dst, ثابت size_t size) ثابت {
        𐡄𐡍 (size > nbytes_remain) {
            أرجع خطأ;
        }
        أرجع read_raw(dst, size) == size;
    }

    uint64_t tell() ثابت {
        أرجع data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ثابت {
        أرجع start_offset;
    }

    منطقي seek(uint64_t absolute_offset) ثابت {
        ثابت uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        𐡄𐡍 (absolute_offset > end_offset) {
            أرجع خطأ;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        أرجع 𐡒𐡔𐡈;
    }

خاص:
    size_t read_raw(فراغ * dst, size_t size) ثابت {
        𐡄𐡍 (callback == nullptr || size == 0) {
            أرجع 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        منطقي reached_eof = خطأ;

        طالما (total_nread < size) {
            ثابت size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            𐡄𐡍 (data_offset + total_nread < data_offset) {
                اكسر;
            }
            ثابت size_t nread = callback(userdata, static_cast<فراغ *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            𐡄𐡍 (nread != chunk_size) {
                reached_eof = 𐡒𐡔𐡈;
                اكسر;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        𐡄𐡍 (reached_eof) {
            nbytes_remain = 0;
        }

        أرجع total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    فراغ * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

بنية gguf_context * gguf_init_empty(فراغ) {
    أرجع جديد gguf_context;
}

template<typename T>
منطقي gguf_read_emplace_helper(ثابت بنية gguf_reader & gr, std::vector<بنية gguf_kv> & kv, ثابت std::string & key, ثابت منطقي is_array, ثابت size_t n) {
    𐡄𐡍 (is_array) {
        std::vector<T> value;
        حاول {
            𐡄𐡍 (!gr.اقرأ(value, n)) {
                أرجع خطأ;
            }
        } امسك (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, key.c_str());
            أرجع خطأ;
        } امسك (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, key.c_str());
            أرجع خطأ;
        }
        kv.emplace_back(key, value);
    } وإلا {
        T value;
        𐡄𐡍 (!gr.اقرأ(value)) {
            أرجع خطأ;
        }
        kv.emplace_back(key, value);
    }
    أرجع 𐡒𐡔𐡈;
}

ساكن بنية gguf_context * gguf_init_from_reader(ثابت بنية gguf_reader & gr, بنية gguf_init_params params) {
    بنية gguf_context * ctx = جديد gguf_context;

    منطقي ok = 𐡒𐡔𐡈;

    // file magic
    {
        std::vector<حرف> magic;
        ok = ok && gr.اقرأ(magic, 4);

        𐡄𐡍 (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            أرجع nullptr;
        }

        لكل (uint32_t i = 0; i < magic.size(); i++) {
            𐡄𐡍 (magic[i] != GGUF_MAGIC[i]) {
                حرف c0 = isprint(magic[0]) ? magic[0] : '?';
                حرف c1 = isprint(magic[1]) ? magic[1] : '?';
                حرف c2 = isprint(magic[2]) ? magic[2] : '?';
                حرف c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                أرجع nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    𐡄𐡍 (ok && gr.اقرأ(ctx->version)) {
        𐡄𐡍 (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = خطأ;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        𐡄𐡍 (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = خطأ;
        }

        𐡄𐡍 (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = خطأ;
        }
        𐡄𐡍 (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = خطأ;
        }
    } وإلا {
        ok = خطأ;
    }

    𐡄𐡍 (ok && gr.اقرأ(n_tensors)) {
        static_assert(حجم(size_t) <= 8 && حجم(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        𐡄𐡍 (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/حجم(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/حجم(gguf_tensor_info));
            ok = خطأ;
        }
    } وإلا {
        ok = خطأ;
    }

    𐡄𐡍 (ok && gr.اقرأ(n_kv)) {
        static_assert(حجم(size_t) <= 8 && حجم(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        𐡄𐡍 (n_kv < 0 || n_kv > int64_t(SIZE_MAX/حجم(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/حجم(gguf_kv));
            ok = خطأ;
        }
    } وإلا {
        ok = خطأ;
    }

    𐡄𐡍 (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        أرجع nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_keys;

        لكل (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            منطقي        is_array = خطأ;
            uint64_t    n        = 1;

            حاول {
                ok = ok && gr.اقرأ(key);
            } امسك (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            } امسك (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            }
            𐡄𐡍 (ok && key.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = خطأ;
            }
            𐡄𐡍 (ok && !seen_keys.insert(key).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, key.c_str(), i);
                ok = خطأ;
            }
            𐡄𐡍 (!ok) {
                اكسر;
            }

            ok = ok && gr.اقرأ(type);
            𐡄𐡍 (type == GGUF_TYPE_ARRAY) {
                is_array = 𐡒𐡔𐡈;
                ok = ok && gr.اقرأ(type);
                ok = ok && gr.اقرأ(n);
            }
            𐡄𐡍 (!ok) {
                اكسر;
            }

            بدل (type) {
                حالة GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<عائم>      (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<منطقي>       (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<مزدوج>     (gr, ctx->kv, key, is_array, n); اكسر;
                حالة GGUF_TYPE_ARRAY:
                افتراضي:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        ok = خطأ;
                    } اكسر;
            }
        }

        𐡄𐡍 (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            أرجع nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ثابت صحيح alignment_idx = gguf_find_key(ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        𐡄𐡍 (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_KEY_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            أرجع nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        𐡄𐡍 (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            أرجع nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    لكل (int64_t i = 0; ok && i < n_tensors; ++i) {
        بنية gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            حاول {
                ok = ok && gr.اقرأ(name);
            } امسك (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            } امسك (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            }
            𐡄𐡍 (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = خطأ;
                اكسر;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            𐡄𐡍 (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = خطأ;
                اكسر;
            }
        }
        𐡄𐡍 (!ok) {
            اكسر;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.اقرأ(n_dims);
            𐡄𐡍 (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = خطأ;
                اكسر;
            }
            لكل (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                𐡄𐡍 (j < n_dims) {
                    ok = ok && gr.اقرأ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                𐡄𐡍 (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = خطأ;
                    اكسر;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            𐡄𐡍 (ok) {
                int64_t current_nelements = info.t.ne[0];
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                    𐡄𐡍 (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        𐡄𐡍 (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = خطأ;
                             اكسر;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 𐡄𐡍 (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     اكسر;
                 }
            }
        }
        𐡄𐡍 (!ok) {
            اكسر;
        }

        // tensor type
        {
            ok = ok && gr.اقرأ(info.t.type);

            // check that tensor type is within defined range
            𐡄𐡍 (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                ok = خطأ;
                اكسر;
            }
            ثابت size_t  type_size = ggml_type_size(info.t.type);
            ثابت int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            𐡄𐡍 (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (صحيح) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = خطأ;
                اكسر;
            }

            // check that the size of the tensor in bytes is representable
            𐡄𐡍 (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = خطأ;
                اكسر;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            لكل (صحيح j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        𐡄𐡍 (!ok) {
            اكسر;
        }

        // tensor data offset within buffer
        ok = ok && gr.اقرأ(info.offset);

        ctx->info.push_back(info);
    }

    𐡄𐡍 (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        أرجع nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    𐡄𐡍 (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        أرجع nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        لكل (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت gguf_tensor_info & ti = ctx->info[i];
            𐡄𐡍 (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                أرجع nullptr;
            }
            ثابت size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            𐡄𐡍 (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                أرجع nullptr;
            }
            ثابت size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            𐡄𐡍 (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                أرجع nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    𐡄𐡍 (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        𐡄𐡍 (params.no_alloc) {
            𐡄𐡍 (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع nullptr;
            }

            ثابت size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } وإلا {
            𐡄𐡍 ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع nullptr;
            }

            ثابت size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            𐡄𐡍 (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        بنية ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        𐡄𐡍 (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            أرجع nullptr;
        }

        بنية ggml_context * ctx_data = *params.ctx;

        بنية ggml_tensor * data = nullptr;

        𐡄𐡍 (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            𐡄𐡍 (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.اقرأ(data->data, ctx->size);

            𐡄𐡍 (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                أرجع nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, 𐡒𐡔𐡈);

        // create the tensors
        لكل (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت بنية gguf_tensor_info & info = ctx->info[i];

            بنية ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            𐡄𐡍 (!ok) {
                اكسر;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            𐡄𐡍 (!params.no_alloc) {
                cur->data = (حرف *) data->data + info.offset;
            }
        }

        𐡄𐡍 (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            أرجع nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    أرجع ctx;
}

بنية gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, فراغ * userdata, size_t max_chunk_read, uint64_t max_expected_size, بنية gguf_init_params params) {
    𐡄𐡍 (callback == nullptr) {
        أرجع nullptr;
    }

    ثابت بنية gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

ساكن size_t gguf_file_reader_callback(فراغ * userdata, فراغ * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    𐡄𐡍 (reader.offset != offset) {
        𐡄𐡍 (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            أرجع 0;
        }

        reader.offset = offset;
    }

    ثابت size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    أرجع nread;
}

بنية gguf_context * gguf_init_from_file_ptr(FILE * file, بنية gguf_init_params params) {
    𐡄𐡍 (!file) {
        أرجع nullptr;
    }

    ثابت int64_t cur = gguf_ftell(file);
    𐡄𐡍 (cur < 0) {
        أرجع nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ثابت بنية gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_buffer_reader {
    ثابت uint8_t * data;
    size_t          size;
};

ساكن size_t gguf_buffer_reader_callback(فراغ * userdata, فراغ * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    ثابت gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    𐡄𐡍 (offset > reader.size || len > reader.size - offset) {
        أرجع 0;
    }

    ثابت size_t data_offset = static_cast<size_t>(offset);
    ثابت size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    أرجع nread;
}

بنية gguf_context * gguf_init_from_buffer(ثابت فراغ * data, size_t size, بنية gguf_init_params params) {
    𐡄𐡍 (data == nullptr || size == 0) {
        أرجع nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<ثابت uint8_t *>(data),
        /*.size = */ size,
    };
    ثابت بنية gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_context * gguf_init_from_file(ثابت حرف * fname, بنية gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    𐡄𐡍 (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        أرجع nullptr;
    }

    بنية gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    أرجع result;
}

فراغ gguf_free(بنية gguf_context * ctx) {
    𐡄𐡍 (ctx == nullptr) {
        أرجع;
    }
    delete ctx;
}

ثابت حرف * gguf_type_name(تعداد gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    أرجع it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(ثابت بنية gguf_context * ctx) {
    أرجع ctx->version;
}

size_t gguf_get_alignment(ثابت بنية gguf_context * ctx) {
    أرجع ctx->alignment;
}

size_t gguf_get_data_offset(ثابت بنية gguf_context * ctx) {
    أرجع ctx->offset;
}

int64_t gguf_get_n_kv(ثابت بنية gguf_context * ctx) {
    أرجع ctx->kv.size();
}

int64_t gguf_find_key(ثابت بنية gguf_context * ctx, ثابت حرف * key) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ثابت int64_t n_kv = gguf_get_n_kv(ctx);

    لكل (int64_t i = 0; i < n_kv; ++i) {
        𐡄𐡍 (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            اكسر;
        }
    }

    أرجع keyfound;
}

ثابت حرف * gguf_get_key(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    أرجع ctx->kv[key_id].get_key().c_str();
}

تعداد gguf_type gguf_get_kv_type(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    أرجع ctx->kv[key_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[key_id].get_type();
}

تعداد gguf_type gguf_get_arr_type(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].is_array);
    أرجع ctx->kv[key_id].get_type();
}

ثابت فراغ * gguf_get_arr_data(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    أرجع ctx->kv[key_id].data.data();
}

ثابت حرف * gguf_get_arr_str(ثابت بنية gguf_context * ctx, int64_t key_id, size_t i) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_STRING);
    أرجع ctx->kv[key_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));

    𐡄𐡍 (ctx->kv[key_id].type == GGUF_TYPE_STRING) {
        أرجع ctx->kv[key_id].data_string.size();
    }

    ثابت size_t type_size = gguf_type_size(ctx->kv[key_id].type);
    GGML_ASSERT(ctx->kv[key_id].data.size() % type_size == 0);
    أرجع ctx->kv[key_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<int32_t>();
}

عائم gguf_get_val_f32(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<عائم>();
}

uint64_t gguf_get_val_u64(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<int64_t>();
}

مزدوج gguf_get_val_f64(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<مزدوج>();
}

منطقي gguf_get_val_bool(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<منطقي>();
}

ثابت حرف * gguf_get_val_str(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    أرجع ctx->kv[key_id].get_val<std::string>().c_str();
}

ثابت فراغ * gguf_get_val_data(ثابت بنية gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    أرجع ctx->kv[key_id].data.data();
}

int64_t gguf_get_n_tensors(ثابت بنية gguf_context * ctx) {
    أرجع ctx->info.size();
}

int64_t gguf_find_tensor(ثابت بنية gguf_context * ctx, ثابت حرف * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    لكل (int64_t i = 0; i < n_tensors; ++i) {
        𐡄𐡍 (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            اكسر;
        }
    }

    أرجع tensor_id;
}

size_t gguf_get_tensor_offset(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ctx->info[tensor_id].offset;
}

ثابت حرف * gguf_get_tensor_name(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ctx->info[tensor_id].t.name;
}

ثابت int64_t * gguf_get_tensor_ne(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ctx->info[tensor_id].t.ne;
}

تعداد ggml_type gguf_get_tensor_type(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_key(بنية gguf_context * ctx, ثابت حرف * key) {
    ثابت int64_t key_id = gguf_find_key(ctx, key);
    𐡄𐡍 (key_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + key_id);
    }
    أرجع key_id;
}

template<typename T>
ساكن فراغ gguf_check_reserved_keys(ثابت std::string & key, ثابت T val) {
    𐡄𐡍 (key == GGUF_KEY_GENERAL_ALIGNMENT) {
        𐡄𐡍 constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_KEY_GENERAL_ALIGNMENT " must be power of 2");
        } وإلا {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_KEY_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

فراغ gguf_set_val_u8(بنية gguf_context * ctx, ثابت حرف * key, uint8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_i8(بنية gguf_context * ctx, ثابت حرف * key, int8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_u16(بنية gguf_context * ctx, ثابت حرف * key, uint16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_i16(بنية gguf_context * ctx, ثابت حرف * key, int16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_u32(بنية gguf_context * ctx, ثابت حرف * key, uint32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_i32(بنية gguf_context * ctx, ثابت حرف * key, int32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_f32(بنية gguf_context * ctx, ثابت حرف * key, عائم val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_u64(بنية gguf_context * ctx, ثابت حرف * key, uint64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_i64(بنية gguf_context * ctx, ثابت حرف * key, int64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_f64(بنية gguf_context * ctx, ثابت حرف * key, مزدوج val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_bool(بنية gguf_context * ctx, ثابت حرف * key, منطقي val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

فراغ gguf_set_val_str(بنية gguf_context * ctx, ثابت حرف * key, ثابت حرف * val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, std::string(val));
}

فراغ gguf_set_arr_data(بنية gguf_context * ctx, ثابت حرف * key, تعداد gguf_type type, ثابت فراغ * data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    ثابت size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    𐡄𐡍 (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(key, tmp);
    ctx->kv.back().cast(type);
}

فراغ gguf_set_arr_str(بنية gguf_context * ctx, ثابت حرف * key, ثابت حرف ** data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    std::vector<std::string> tmp(n);
    لكل (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(key, tmp);
}

// set or add KV pairs from another context
فراغ gguf_set_kv(بنية gguf_context * ctx, ثابت بنية gguf_context * src) {
    ثابت int64_t n_kv = gguf_get_n_kv(src);
    لكل (int64_t i = 0; i < n_kv; ++i) {
        ثابت بنية gguf_kv & kv = src->kv[i];

        𐡄𐡍 (!kv.is_array) {
            بدل (kv.get_type()) {
                حالة GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_key().c_str(), kv.get_val<uint8_t>());             اكسر;
                حالة GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_key().c_str(), kv.get_val<int8_t>());              اكسر;
                حالة GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_key().c_str(), kv.get_val<uint16_t>());            اكسر;
                حالة GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_key().c_str(), kv.get_val<int16_t>());             اكسر;
                حالة GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_key().c_str(), kv.get_val<uint32_t>());            اكسر;
                حالة GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_key().c_str(), kv.get_val<int32_t>());             اكسر;
                حالة GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_key().c_str(), kv.get_val<عائم>());               اكسر;
                حالة GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_key().c_str(), kv.get_val<uint64_t>());            اكسر;
                حالة GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_key().c_str(), kv.get_val<int64_t>());             اكسر;
                حالة GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_key().c_str(), kv.get_val<مزدوج>());              اكسر;
                حالة GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_key().c_str(), kv.get_val<منطقي>());                اكسر;
                حالة GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_key().c_str(), kv.get_val<std::string>().c_str()); اكسر;
                حالة GGUF_TYPE_ARRAY:
                افتراضي: GGML_ABORT("invalid type");
            }
            استمر;
        }

        ثابت size_t ne = kv.get_ne();

        بدل (kv.get_type()) {
            حالة GGUF_TYPE_UINT8:
            حالة GGUF_TYPE_INT8:
            حالة GGUF_TYPE_UINT16:
            حالة GGUF_TYPE_INT16:
            حالة GGUF_TYPE_UINT32:
            حالة GGUF_TYPE_INT32:
            حالة GGUF_TYPE_FLOAT32:
            حالة GGUF_TYPE_UINT64:
            حالة GGUF_TYPE_INT64:
            حالة GGUF_TYPE_FLOAT64:
            حالة GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_key().c_str(), kv.get_type(), kv.data.data(), ne);
            } اكسر;
            حالة GGUF_TYPE_STRING: {
                std::vector<ثابت حرف *> tmp(ne);
                لكل (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_key().c_str(), tmp.data(), ne);
            } اكسر;
            حالة GGUF_TYPE_ARRAY:
            افتراضي: GGML_ABORT("invalid type");
        }
    }
}

فراغ gguf_add_tensor(
             بنية gguf_context * ctx,
        ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    𐡄𐡍 (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    بنية gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

فراغ gguf_set_tensor_type(بنية gguf_context * ctx, ثابت حرف * name, تعداد ggml_type type) {
    ثابت int64_t tensor_id = gguf_find_tensor(ctx, name);
    𐡄𐡍 (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    بنية ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ثابت size_t  type_size = ggml_type_size(type);
    ثابت int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);
    لكل (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

فراغ gguf_set_tensor_data(بنية gguf_context * ctx, ثابت حرف * name, ثابت فراغ * data) {
    ثابت int64_t tensor_id = gguf_find_tensor(ctx, name);
    𐡄𐡍 (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (فراغ *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

بنية gguf_writer_base {
    size_t written_bytes {0u};

    virtual ~gguf_writer_base(فراغ) = افتراضي;

    // we bet on devirtualization
    virtual فراغ اكتب(int8_t val) = 0;
    virtual فراغ اكتب(ثابت std::vector<int8_t> & val) = 0;
    virtual فراغ write_tensor_data(ثابت بنية gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    فراغ اكتب(ثابت T & val) {
        لكل (size_t i = 0; i < حجم(val); ++i) {
            اكتب(reinterpret_cast<ثابت int8_t *>(&val)[i]);
        }
    }

    فراغ اكتب(ثابت منطقي & val) {
        ثابت int8_t val8 = val ? 1 : 0;
        اكتب(val8);
    }

    فراغ اكتب(ثابت std::string & val) {
        {
            ثابت uint64_t n = val.length();
            اكتب(n);
        }
        لكل (size_t i = 0; i < val.length(); ++i) {
            اكتب((val.data())[i]);
        }
    }

    فراغ اكتب(ثابت حرف * val) {
        اكتب(std::string(val));
    }

    فراغ اكتب(ثابت تعداد ggml_type & val) {
        اكتب(int32_t(val));
    }

    فراغ اكتب(ثابت تعداد gguf_type & val) {
        اكتب(int32_t(val));
    }

    فراغ اكتب(ثابت بنية gguf_kv & kv) {
        ثابت uint64_t ne = kv.get_ne();

        اكتب(kv.get_key());

        𐡄𐡍 (kv.is_array) {
            اكتب(GGUF_TYPE_ARRAY);
            اكتب(kv.get_type());
            اكتب(ne);
        } وإلا {
            اكتب(kv.get_type());
        }

        بدل (kv.get_type()) {
            حالة GGUF_TYPE_UINT8:
            حالة GGUF_TYPE_INT8:
            حالة GGUF_TYPE_UINT16:
            حالة GGUF_TYPE_INT16:
            حالة GGUF_TYPE_UINT32:
            حالة GGUF_TYPE_INT32:
            حالة GGUF_TYPE_FLOAT32:
            حالة GGUF_TYPE_UINT64:
            حالة GGUF_TYPE_INT64:
            حالة GGUF_TYPE_FLOAT64: {
                اكتب(kv.data);
            } اكسر;
            حالة GGUF_TYPE_BOOL: {
                لكل (size_t i = 0; i < ne; ++i) {
                    اكتب(kv.get_val<منطقي>(i));
                }
            } اكسر;
            حالة GGUF_TYPE_STRING: {
                لكل (size_t i = 0; i < ne; ++i) {
                    اكتب(kv.get_val<std::string>(i));
                }
            } اكسر;
            حالة GGUF_TYPE_ARRAY:
            افتراضي: GGML_ABORT("invalid type");
        }
    }

    فراغ write_tensor_meta(ثابت بنية gguf_tensor_info & info) {
        اكتب(info.t.name);

        ثابت uint32_t n_dims = ggml_n_dims(&info.t);
        اكتب(n_dims);

        لكل (uint32_t j = 0; j < n_dims; ++j) {
            اكتب(info.t.ne[j]);
        }
        اكتب(info.t.type);
        اكتب(info.offset);
    }

    فراغ pad(ثابت size_t alignment) {
        طالما (written_bytes % alignment != 0) {
            ثابت int8_t zero = 0;
            اكتب(zero);
        }
    }
};

// vector buffer based writer
بنية gguf_writer_buf final : عام gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::اكتب;

    فراغ اكتب(ثابت int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    فراغ اكتب(ثابت std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    فراغ write_tensor_data(ثابت بنية gguf_tensor_info & info, ثابت size_t offset_data, ثابت size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t offset = buf.size();
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        𐡄𐡍 (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } وإلا {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
بنية gguf_writer_file final : عام gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::اكتب;

    فراغ اكتب(ثابت int8_t val) override {
        ثابت auto real_val = static_cast<uint8_t>(val);
        ثابت auto ret = fputc(real_val, file);
        written_bytes++;
        𐡄𐡍 (ret != real_val) {
            ارم std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((صحيح)real_val) + "'");
        }
    }

    فراغ اكتب(ثابت std::vector<int8_t> & val) override {
        ثابت auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        𐡄𐡍 (ret != val.size()) {
            ارم std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    فراغ write_tensor_data(ثابت بنية gguf_tensor_info & info, ثابت size_t offset_data, ثابت size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        𐡄𐡍 (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } وإلا {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        اكتب(buf);

        pad(alignment);
    }
};

template <typename writer_t>
ساكن فراغ gguf_write_out(ثابت بنية gguf_context * ctx, writer_t & gw, منطقي only_meta) {
    ثابت int64_t n_kv      = gguf_get_n_kv(ctx);
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.اكتب(GGUF_MAGIC[0]);
    gw.اكتب(GGUF_MAGIC[1]);
    gw.اكتب(GGUF_MAGIC[2]);
    gw.اكتب(GGUF_MAGIC[3]);
    gw.اكتب(ctx->version);
    gw.اكتب(n_tensors);
    gw.اكتب(n_kv);

    // write key-value pairs
    لكل (int64_t i = 0; i < n_kv; ++i) {
        gw.اكتب(ctx->kv[i]);
    }

    // write tensor info
    لكل (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    𐡄𐡍 (only_meta) {
        أرجع;
    }

    ثابت size_t offset_data = gw.written_bytes;

    // write tensor data
    لكل (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

فراغ gguf_write_to_buf(ثابت بنية gguf_context * ctx, std::vector<int8_t> & buf, منطقي only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

منطقي gguf_write_to_file_ptr(ثابت بنية gguf_context * ctx, FILE * file, منطقي only_meta) {
    GGML_ASSERT(file);

    حاول {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } امسك (ثابت std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        أرجع خطأ;
    }
    أرجع 𐡒𐡔𐡈;
}

منطقي gguf_write_to_file(ثابت بنية gguf_context * ctx, ثابت حرف * fname, منطقي only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    𐡄𐡍 (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        أرجع خطأ;
    }

    ثابت منطقي success = gguf_write_to_file_ptr(ctx, file, only_meta);
    𐡄𐡍 (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    أرجع success;
}

size_t gguf_get_meta_size(ثابت بنية gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ 𐡒𐡔𐡈);
    أرجع buf.size();
}

فراغ gguf_get_meta_data(ثابت بنية gguf_context * ctx, فراغ * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ 𐡒𐡔𐡈);
    memcpy(data, buf.data(), buf.size());
}
