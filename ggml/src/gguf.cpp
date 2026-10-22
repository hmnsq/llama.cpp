#شامل "ggml.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "gguf.h"

#شامل <cinttypes>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdio>
#شامل <cstdlib>
#شامل <cstring>
#شامل <cerrno>
#شامل <map>
#شامل <new>
#شامل <stdexcept>
#شامل <string>
#شامل <unordered_set>
#شامل <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#وگرنه
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
ساختار type_to_gguf_type;

template <>
ساختار type_to_gguf_type<uint8_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_UINT8;
};

template <>
ساختار type_to_gguf_type<int8_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_INT8;
};

template <>
ساختار type_to_gguf_type<uint16_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_UINT16;
};

template <>
ساختار type_to_gguf_type<int16_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_INT16;
};

template <>
ساختار type_to_gguf_type<uint32_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_UINT32;
};

template <>
ساختار type_to_gguf_type<int32_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_INT32;
};

template <>
ساختار type_to_gguf_type<اعشاری> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
ساختار type_to_gguf_type<منطقی> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_BOOL;
};

template <>
ساختار type_to_gguf_type<std::string> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_STRING;
};

template <>
ساختار type_to_gguf_type<uint64_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_UINT64;
};

template <>
ساختار type_to_gguf_type<int64_t> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_INT64;
};

template <>
ساختار type_to_gguf_type<دوگانه> {
    ایستا constexpr شمارش gguf_type value = GGUF_TYPE_FLOAT64;
};

ایستا ثابت std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   اندازه(uint8_t)},
    {GGUF_TYPE_INT8,    اندازه(int8_t)},
    {GGUF_TYPE_UINT16,  اندازه(uint16_t)},
    {GGUF_TYPE_INT16,   اندازه(int16_t)},
    {GGUF_TYPE_UINT32,  اندازه(uint32_t)},
    {GGUF_TYPE_INT32,   اندازه(int32_t)},
    {GGUF_TYPE_FLOAT32, اندازه(اعشاری)},
    {GGUF_TYPE_BOOL,    اندازه(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  اندازه(uint64_t)},
    {GGUF_TYPE_INT64,   اندازه(int64_t)},
    {GGUF_TYPE_FLOAT64, اندازه(دوگانه)},
};
static_assert(GGUF_TYPE_شمار == 13, "GGUF_TYPE_COUNT != 13");

ایستا ثابت std::map<gguf_type, ثابت نویسه *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_شمار == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(شمارش gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    بازگشت it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

ساختار gguf_kv {
    std::string کلید;

    منطقی is_array;
    شمارش gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(ثابت std::string & کلید, ثابت T value)
            : کلید(کلید), is_array(نادرست), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!کلید.empty());
        data.resize(اندازه(T));
        memcpy(data.data(), &value, اندازه(T));
    }

    template <typename T>
    gguf_kv(ثابت std::string & کلید, ثابت std::vector<T> & value)
            : کلید(کلید), is_array(درست), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!کلید.empty());
        data.resize(value.size()*اندازه(T));
        برای (size_t i = 0; i < value.size(); ++i) {
            ثابت T tmp = value[i];
            memcpy(data.data() + i*اندازه(T), &tmp, اندازه(T));
        }
    }

    gguf_kv(ثابت std::string & کلید, ثابت std::string & value)
            : کلید(کلید), is_array(نادرست), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلید.empty());
        data_string.push_back(value);
    }

    gguf_kv(ثابت std::string & کلید, ثابت std::vector<std::string> & value)
            : کلید(کلید), is_array(درست), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلید.empty());
        data_string = value;
    }

    ثابت std::string & get_کلید() ثابت {
        بازگشت کلید;
    }

    ثابت شمارش gguf_type & get_type() ثابت {
        بازگشت type;
    }

    size_t get_ne() ثابت {
        اگر (type == GGUF_TYPE_STRING) {
            ثابت size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            بازگشت ne;
        }
        ثابت size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        ثابت size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        بازگشت ne;
    }

    template <typename T>
    ثابت T & get_val(ثابت size_t i = 0) ثابت {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        اگر constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            بازگشت data_string[i];
        }
        ثابت size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        بازگشت reinterpret_cast<ثابت T *>(data.data())[i];
    }

    تهی cast(ثابت شمارش gguf_type new_type) {
        ثابت size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

ساختار gguf_tensor_info {
    ساختار ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

ساختار gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<ساختار gguf_kv> kv;
    std::vector<ساختار gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    تهی * data = nullptr;
};

ساختار gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            تهی * userdata,
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
    ایستا uint64_t file_remain(FILE * file) {
        ثابت int64_t cur = gguf_ftell(file);
        اگر (cur < 0) {
            بازگشت 0;
        }
        اگر (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            بازگشت 0;
        }
        ثابت int64_t end = gguf_ftell(file);
        اگر (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            بازگشت 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        بازگشت static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    منطقی اقرأ(T & dst) ثابت {
        ثابت size_t size = اندازه(dst);
        اگر (size > nbytes_remain) {
            بازگشت نادرست;
        }
        بازگشت read_raw(&dst, size) == size;
    }

    template <typename T>
    منطقی اقرأ(std::vector<T> & dst, ثابت size_t n) ثابت {
        اگر (n > GGUF_MAX_ARRAY_ELEMENTS) {
            بازگشت نادرست;
        }
        اگر constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            اگر (n > SIZE_MAX / اندازه(uint64_t)) {
                بازگشت نادرست;
            }
            اگر (nbytes_remain < n * اندازه(uint64_t)) {
                بازگشت نادرست;
            }
        } وگرنه {
            اگر (n > SIZE_MAX / اندازه(T)) {
                بازگشت نادرست;
            }
            اگر (nbytes_remain < n * اندازه(T)) {
                بازگشت نادرست;
            }
        }
        dst.resize(n);
        برای (size_t i = 0; i < dst.size(); ++i) {
            اگر constexpr (std::is_same<T, منطقی>::value) {
                منطقی tmp;
                اگر (!اقرأ(tmp)) {
                    بازگشت نادرست;
                }
                dst[i] = tmp;
            } وگرنه {
                اگر (!اقرأ(dst[i])) {
                    بازگشت نادرست;
                }
            }
        }
        بازگشت درست;
    }

    منطقی اقرأ(منطقی & dst) ثابت {
        int8_t tmp = -1;
        اگر (!اقرأ(tmp)) {
            بازگشت نادرست;
        }
        dst = tmp != 0;
        بازگشت درست;
    }

    منطقی اقرأ(شمارش ggml_type & dst) ثابت {
        int32_t tmp = -1;
        اگر (!اقرأ(tmp)) {
            بازگشت نادرست;
        }
        dst = ggml_type(tmp);
        بازگشت درست;
    }

    منطقی اقرأ(شمارش gguf_type & dst) ثابت {
        int32_t tmp = -1;
        اگر (!اقرأ(tmp)) {
            بازگشت نادرست;
        }
        dst = gguf_type(tmp);
        بازگشت درست;
    }

    منطقی اقرأ(std::string & dst) ثابت {
        uint64_t size = 0;
        اگر (!اقرأ(size)) {
            بازگشت نادرست;
        }
        اگر (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            بازگشت نادرست;
        }
        اگر (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            بازگشت نادرست;
        }
        dst.resize(static_cast<size_t>(size));
        بازگشت read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    منطقی اقرأ(تهی * dst, ثابت size_t size) ثابت {
        اگر (size > nbytes_remain) {
            بازگشت نادرست;
        }
        بازگشت read_raw(dst, size) == size;
    }

    uint64_t tell() ثابت {
        بازگشت data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ثابت {
        بازگشت start_offset;
    }

    منطقی seek(uint64_t absolute_offset) ثابت {
        ثابت uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        اگر (absolute_offset > end_offset) {
            بازگشت نادرست;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        بازگشت درست;
    }

خصوصی:
    size_t read_raw(تهی * dst, size_t size) ثابت {
        اگر (callback == nullptr || size == 0) {
            بازگشت 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        منطقی reached_eof = نادرست;

        وقتی (total_nread < size) {
            ثابت size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            اگر (data_offset + total_nread < data_offset) {
                بشکن;
            }
            ثابت size_t nread = callback(userdata, static_cast<تهی *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            اگر (nread != chunk_size) {
                reached_eof = درست;
                بشکن;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        اگر (reached_eof) {
            nbytes_remain = 0;
        }

        بازگشت total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    تهی * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

ساختار gguf_context * gguf_init_empty(تهی) {
    بازگشت جدید gguf_context;
}

template<typename T>
منطقی gguf_read_emplace_helper(ثابت ساختار gguf_reader & gr, std::vector<ساختار gguf_kv> & kv, ثابت std::string & کلید, ثابت منطقی is_array, ثابت size_t n) {
    اگر (is_array) {
        std::vector<T> value;
        تلاش {
            اگر (!gr.اقرأ(value, n)) {
                بازگشت نادرست;
            }
        } بگیر (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, کلید.c_str());
            بازگشت نادرست;
        } بگیر (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, کلید.c_str());
            بازگشت نادرست;
        }
        kv.emplace_back(کلید, value);
    } وگرنه {
        T value;
        اگر (!gr.اقرأ(value)) {
            بازگشت نادرست;
        }
        kv.emplace_back(کلید, value);
    }
    بازگشت درست;
}

ایستا ساختار gguf_context * gguf_init_from_reader(ثابت ساختار gguf_reader & gr, ساختار gguf_init_params params) {
    ساختار gguf_context * ctx = جدید gguf_context;

    منطقی ok = درست;

    // file magic
    {
        std::vector<نویسه> magic;
        ok = ok && gr.اقرأ(magic, 4);

        اگر (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            بازگشت nullptr;
        }

        برای (uint32_t i = 0; i < magic.size(); i++) {
            اگر (magic[i] != GGUF_MAGIC[i]) {
                نویسه c0 = isprint(magic[0]) ? magic[0] : '?';
                نویسه c1 = isprint(magic[1]) ? magic[1] : '?';
                نویسه c2 = isprint(magic[2]) ? magic[2] : '?';
                نویسه c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                بازگشت nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    اگر (ok && gr.اقرأ(ctx->version)) {
        اگر (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = نادرست;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        اگر (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = نادرست;
        }

        اگر (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = نادرست;
        }
        اگر (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = نادرست;
        }
    } وگرنه {
        ok = نادرست;
    }

    اگر (ok && gr.اقرأ(n_tensors)) {
        static_assert(اندازه(size_t) <= 8 && اندازه(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        اگر (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/اندازه(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/اندازه(gguf_tensor_info));
            ok = نادرست;
        }
    } وگرنه {
        ok = نادرست;
    }

    اگر (ok && gr.اقرأ(n_kv)) {
        static_assert(اندازه(size_t) <= 8 && اندازه(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        اگر (n_kv < 0 || n_kv > int64_t(SIZE_MAX/اندازه(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/اندازه(gguf_kv));
            ok = نادرست;
        }
    } وگرنه {
        ok = نادرست;
    }

    اگر (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        بازگشت nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_کلیدها;

        برای (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string کلید;
            gguf_type   type     = gguf_type(-1);
            منطقی        is_array = نادرست;
            uint64_t    n        = 1;

            تلاش {
                ok = ok && gr.اقرأ(کلید);
            } بگیر (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = نادرست;
            } بگیر (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = نادرست;
            }
            اگر (ok && کلید.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = نادرست;
            }
            اگر (ok && !seen_کلیدها.insert(کلید).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, کلید.c_str(), i);
                ok = نادرست;
            }
            اگر (!ok) {
                بشکن;
            }

            ok = ok && gr.اقرأ(type);
            اگر (type == GGUF_TYPE_ARRAY) {
                is_array = درست;
                ok = ok && gr.اقرأ(type);
                ok = ok && gr.اقرأ(n);
            }
            اگر (!ok) {
                بشکن;
            }

            انتخاب (type) {
                حالت GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<اعشاری>      (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<منطقی>       (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<دوگانه>     (gr, ctx->kv, کلید, is_array, n); بشکن;
                حالت GGUF_TYPE_ARRAY:
                پیشفرض:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, کلید.c_str(), type);
                        ok = نادرست;
                    } بشکن;
            }
        }

        اگر (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            بازگشت nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ثابت صحیح alignment_idx = gguf_find_کلید(ctx, GGUF_کلید_GENERAL_ALIGNMENT);
        اگر (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_کلید_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            بازگشت nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        اگر (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            بازگشت nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    برای (int64_t i = 0; ok && i < n_tensors; ++i) {
        ساختار gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            تلاش {
                ok = ok && gr.اقرأ(name);
            } بگیر (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = نادرست;
            } بگیر (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = نادرست;
            }
            اگر (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = نادرست;
                بشکن;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            اگر (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = نادرست;
                بشکن;
            }
        }
        اگر (!ok) {
            بشکن;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.اقرأ(n_dims);
            اگر (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = نادرست;
                بشکن;
            }
            برای (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                اگر (j < n_dims) {
                    ok = ok && gr.اقرأ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                اگر (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = نادرست;
                    بشکن;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            اگر (ok) {
                int64_t current_nelements = info.t.ne[0];
                برای (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
                    اگر (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        اگر (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = نادرست;
                             بشکن;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 اگر (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     بشکن;
                 }
            }
        }
        اگر (!ok) {
            بشکن;
        }

        // tensor type
        {
            ok = ok && gr.اقرأ(info.t.type);

            // check that tensor type is within defined range
            اگر (info.t.type < 0 || info.t.type >= GGML_TYPE_شمار) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_شمار);
                ok = نادرست;
                بشکن;
            }
            ثابت size_t  type_size = ggml_type_size(info.t.type);
            ثابت int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            اگر (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (صحیح) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = نادرست;
                بشکن;
            }

            // check that the size of the tensor in bytes is representable
            اگر (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = نادرست;
                بشکن;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            برای (صحیح j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        اگر (!ok) {
            بشکن;
        }

        // tensor data offset within buffer
        ok = ok && gr.اقرأ(info.offset);

        ctx->info.push_back(info);
    }

    اگر (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        بازگشت nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    اگر (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        بازگشت nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        برای (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت gguf_tensor_info & ti = ctx->info[i];
            اگر (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                بازگشت nullptr;
            }
            ثابت size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            اگر (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                بازگشت nullptr;
            }
            ثابت size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            اگر (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                بازگشت nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    اگر (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        اگر (params.no_alloc) {
            اگر (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                بازگشت nullptr;
            }

            ثابت size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } وگرنه {
            اگر ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                بازگشت nullptr;
            }

            ثابت size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            اگر (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                بازگشت nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        ساختار ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        اگر (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            بازگشت nullptr;
        }

        ساختار ggml_context * ctx_data = *params.ctx;

        ساختار ggml_tensor * data = nullptr;

        اگر (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            اگر (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.اقرأ(data->data, ctx->size);

            اگر (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                بازگشت nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, درست);

        // create the tensors
        برای (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت ساختار gguf_tensor_info & info = ctx->info[i];

            ساختار ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            اگر (!ok) {
                بشکن;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            اگر (!params.no_alloc) {
                cur->data = (نویسه *) data->data + info.offset;
            }
        }

        اگر (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            بازگشت nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    بازگشت ctx;
}

ساختار gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, تهی * userdata, size_t max_chunk_read, uint64_t max_expected_size, ساختار gguf_init_params params) {
    اگر (callback == nullptr) {
        بازگشت nullptr;
    }

    ثابت ساختار gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    بازگشت gguf_init_from_reader(gr, params);
}

ساختار gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

ایستا size_t gguf_file_reader_callback(تهی * userdata, تهی * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    اگر (reader.offset != offset) {
        اگر (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            بازگشت 0;
        }

        reader.offset = offset;
    }

    ثابت size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    بازگشت nread;
}

ساختار gguf_context * gguf_init_from_file_ptr(FILE * file, ساختار gguf_init_params params) {
    اگر (!file) {
        بازگشت nullptr;
    }

    ثابت int64_t cur = gguf_ftell(file);
    اگر (cur < 0) {
        بازگشت nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ثابت ساختار gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    بازگشت gguf_init_from_reader(gr, params);
}

ساختار gguf_buffer_reader {
    ثابت uint8_t * data;
    size_t          size;
};

ایستا size_t gguf_buffer_reader_callback(تهی * userdata, تهی * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    ثابت gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    اگر (offset > reader.size || len > reader.size - offset) {
        بازگشت 0;
    }

    ثابت size_t data_offset = static_cast<size_t>(offset);
    ثابت size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    بازگشت nread;
}

ساختار gguf_context * gguf_init_from_buffer(ثابت تهی * data, size_t size, ساختار gguf_init_params params) {
    اگر (data == nullptr || size == 0) {
        بازگشت nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<ثابت uint8_t *>(data),
        /*.size = */ size,
    };
    ثابت ساختار gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    بازگشت gguf_init_from_reader(gr, params);
}

ساختار gguf_context * gguf_init_from_file(ثابت نویسه * fname, ساختار gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    اگر (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        بازگشت nullptr;
    }

    ساختار gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    بازگشت result;
}

تهی gguf_free(ساختار gguf_context * ctx) {
    اگر (ctx == nullptr) {
        بازگشت;
    }
    delete ctx;
}

ثابت نویسه * gguf_type_name(شمارش gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    بازگشت it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(ثابت ساختار gguf_context * ctx) {
    بازگشت ctx->version;
}

size_t gguf_get_alignment(ثابت ساختار gguf_context * ctx) {
    بازگشت ctx->alignment;
}

size_t gguf_get_data_offset(ثابت ساختار gguf_context * ctx) {
    بازگشت ctx->offset;
}

int64_t gguf_get_n_kv(ثابت ساختار gguf_context * ctx) {
    بازگشت ctx->kv.size();
}

int64_t gguf_find_کلید(ثابت ساختار gguf_context * ctx, ثابت نویسه * کلید) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ثابت int64_t n_kv = gguf_get_n_kv(ctx);

    برای (int64_t i = 0; i < n_kv; ++i) {
        اگر (strcmp(کلید, gguf_get_کلید(ctx, i)) == 0) {
            keyfound = i;
            بشکن;
        }
    }

    بازگشت keyfound;
}

ثابت نویسه * gguf_get_کلید(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    بازگشت ctx->kv[کلید_id].get_کلید().c_str();
}

شمارش gguf_type gguf_get_kv_type(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    بازگشت ctx->kv[کلید_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[کلید_id].get_type();
}

شمارش gguf_type gguf_get_arr_type(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].is_array);
    بازگشت ctx->kv[کلید_id].get_type();
}

ثابت تهی * gguf_get_arr_data(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_type() != GGUF_TYPE_STRING);
    بازگشت ctx->kv[کلید_id].data.data();
}

ثابت نویسه * gguf_get_arr_str(ثابت ساختار gguf_context * ctx, int64_t کلید_id, size_t i) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_type() == GGUF_TYPE_STRING);
    بازگشت ctx->kv[کلید_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));

    اگر (ctx->kv[کلید_id].type == GGUF_TYPE_STRING) {
        بازگشت ctx->kv[کلید_id].data_string.size();
    }

    ثابت size_t type_size = gguf_type_size(ctx->kv[کلید_id].type);
    GGML_ASSERT(ctx->kv[کلید_id].data.size() % type_size == 0);
    بازگشت ctx->kv[کلید_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<int32_t>();
}

اعشاری gguf_get_val_f32(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<اعشاری>();
}

uint64_t gguf_get_val_u64(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<int64_t>();
}

دوگانه gguf_get_val_f64(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<دوگانه>();
}

منطقی gguf_get_val_bool(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<منطقی>();
}

ثابت نویسه * gguf_get_val_str(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    بازگشت ctx->kv[کلید_id].get_val<std::string>().c_str();
}

ثابت تهی * gguf_get_val_data(ثابت ساختار gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[کلید_id].get_type() != GGUF_TYPE_STRING);
    بازگشت ctx->kv[کلید_id].data.data();
}

int64_t gguf_get_n_tensors(ثابت ساختار gguf_context * ctx) {
    بازگشت ctx->info.size();
}

int64_t gguf_find_tensor(ثابت ساختار gguf_context * ctx, ثابت نویسه * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    برای (int64_t i = 0; i < n_tensors; ++i) {
        اگر (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            بشکن;
        }
    }

    بازگشت tensor_id;
}

size_t gguf_get_tensor_offset(ثابت ساختار gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    بازگشت ctx->info[tensor_id].offset;
}

ثابت نویسه * gguf_get_tensor_name(ثابت ساختار gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    بازگشت ctx->info[tensor_id].t.name;
}

ثابت int64_t * gguf_get_tensor_ne(ثابت ساختار gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    بازگشت ctx->info[tensor_id].t.ne;
}

شمارش ggml_type gguf_get_tensor_type(ثابت ساختار gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    بازگشت ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(ثابت ساختار gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    بازگشت ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_کلید(ساختار gguf_context * ctx, ثابت نویسه * کلید) {
    ثابت int64_t کلید_id = gguf_find_کلید(ctx, کلید);
    اگر (کلید_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + کلید_id);
    }
    بازگشت کلید_id;
}

template<typename T>
ایستا تهی gguf_check_reserved_کلیدها(ثابت std::string & کلید, ثابت T val) {
    اگر (کلید == GGUF_کلید_GENERAL_ALIGNMENT) {
        اگر constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_کلید_GENERAL_ALIGNMENT " must be power of 2");
        } وگرنه {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_کلید_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

تهی gguf_set_val_u8(ساختار gguf_context * ctx, ثابت نویسه * کلید, uint8_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_i8(ساختار gguf_context * ctx, ثابت نویسه * کلید, int8_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_u16(ساختار gguf_context * ctx, ثابت نویسه * کلید, uint16_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_i16(ساختار gguf_context * ctx, ثابت نویسه * کلید, int16_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_u32(ساختار gguf_context * ctx, ثابت نویسه * کلید, uint32_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_i32(ساختار gguf_context * ctx, ثابت نویسه * کلید, int32_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_f32(ساختار gguf_context * ctx, ثابت نویسه * کلید, اعشاری val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_u64(ساختار gguf_context * ctx, ثابت نویسه * کلید, uint64_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_i64(ساختار gguf_context * ctx, ثابت نویسه * کلید, int64_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_f64(ساختار gguf_context * ctx, ثابت نویسه * کلید, دوگانه val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_bool(ساختار gguf_context * ctx, ثابت نویسه * کلید, منطقی val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

تهی gguf_set_val_str(ساختار gguf_context * ctx, ثابت نویسه * کلید, ثابت نویسه * val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, std::string(val));
}

تهی gguf_set_arr_data(ساختار gguf_context * ctx, ثابت نویسه * کلید, شمارش gguf_type type, ثابت تهی * data, size_t n) {
    gguf_check_reserved_کلیدها(کلید, data);
    gguf_remove_کلید(ctx, کلید);

    ثابت size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    اگر (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(کلید, tmp);
    ctx->kv.back().cast(type);
}

تهی gguf_set_arr_str(ساختار gguf_context * ctx, ثابت نویسه * کلید, ثابت نویسه ** data, size_t n) {
    gguf_check_reserved_کلیدها(کلید, data);
    gguf_remove_کلید(ctx, کلید);

    std::vector<std::string> tmp(n);
    برای (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(کلید, tmp);
}

// set or add KV pairs from another context
تهی gguf_set_kv(ساختار gguf_context * ctx, ثابت ساختار gguf_context * src) {
    ثابت int64_t n_kv = gguf_get_n_kv(src);
    برای (int64_t i = 0; i < n_kv; ++i) {
        ثابت ساختار gguf_kv & kv = src->kv[i];

        اگر (!kv.is_array) {
            انتخاب (kv.get_type()) {
                حالت GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_کلید().c_str(), kv.get_val<uint8_t>());             بشکن;
                حالت GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_کلید().c_str(), kv.get_val<int8_t>());              بشکن;
                حالت GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_کلید().c_str(), kv.get_val<uint16_t>());            بشکن;
                حالت GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_کلید().c_str(), kv.get_val<int16_t>());             بشکن;
                حالت GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_کلید().c_str(), kv.get_val<uint32_t>());            بشکن;
                حالت GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_کلید().c_str(), kv.get_val<int32_t>());             بشکن;
                حالت GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_کلید().c_str(), kv.get_val<اعشاری>());               بشکن;
                حالت GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_کلید().c_str(), kv.get_val<uint64_t>());            بشکن;
                حالت GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_کلید().c_str(), kv.get_val<int64_t>());             بشکن;
                حالت GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_کلید().c_str(), kv.get_val<دوگانه>());              بشکن;
                حالت GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_کلید().c_str(), kv.get_val<منطقی>());                بشکن;
                حالت GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_کلید().c_str(), kv.get_val<std::string>().c_str()); بشکن;
                حالت GGUF_TYPE_ARRAY:
                پیشفرض: GGML_ABORT("invalid type");
            }
            ادامه;
        }

        ثابت size_t ne = kv.get_ne();

        انتخاب (kv.get_type()) {
            حالت GGUF_TYPE_UINT8:
            حالت GGUF_TYPE_INT8:
            حالت GGUF_TYPE_UINT16:
            حالت GGUF_TYPE_INT16:
            حالت GGUF_TYPE_UINT32:
            حالت GGUF_TYPE_INT32:
            حالت GGUF_TYPE_FLOAT32:
            حالت GGUF_TYPE_UINT64:
            حالت GGUF_TYPE_INT64:
            حالت GGUF_TYPE_FLOAT64:
            حالت GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_کلید().c_str(), kv.get_type(), kv.data.data(), ne);
            } بشکن;
            حالت GGUF_TYPE_STRING: {
                std::vector<ثابت نویسه *> tmp(ne);
                برای (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_کلید().c_str(), tmp.data(), ne);
            } بشکن;
            حالت GGUF_TYPE_ARRAY:
            پیشفرض: GGML_ABORT("invalid type");
        }
    }
}

تهی gguf_add_tensor(
             ساختار gguf_context * ctx,
        ثابت ساختار ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    اگر (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    ساختار gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

تهی gguf_set_tensor_type(ساختار gguf_context * ctx, ثابت نویسه * name, شمارش ggml_type type) {
    ثابت int64_t tensor_id = gguf_find_tensor(ctx, name);
    اگر (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    ساختار ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ثابت size_t  type_size = ggml_type_size(type);
    ثابت int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    برای (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);
    برای (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

تهی gguf_set_tensor_data(ساختار gguf_context * ctx, ثابت نویسه * name, ثابت تهی * data) {
    ثابت int64_t tensor_id = gguf_find_tensor(ctx, name);
    اگر (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (تهی *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

ساختار gguf_writer_base {
    size_t written_bytes {0u};

    مجازی ~gguf_writer_base(تهی) = پیشفرض;

    // we bet on devirtualization
    مجازی تهی اكتب(int8_t val) = 0;
    مجازی تهی اكتب(ثابت std::vector<int8_t> & val) = 0;
    مجازی تهی write_tensor_data(ثابت ساختار gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    تهی اكتب(ثابت T & val) {
        برای (size_t i = 0; i < اندازه(val); ++i) {
            اكتب(reinterpret_cast<ثابت int8_t *>(&val)[i]);
        }
    }

    تهی اكتب(ثابت منطقی & val) {
        ثابت int8_t val8 = val ? 1 : 0;
        اكتب(val8);
    }

    تهی اكتب(ثابت std::string & val) {
        {
            ثابت uint64_t n = val.length();
            اكتب(n);
        }
        برای (size_t i = 0; i < val.length(); ++i) {
            اكتب((val.data())[i]);
        }
    }

    تهی اكتب(ثابت نویسه * val) {
        اكتب(std::string(val));
    }

    تهی اكتب(ثابت شمارش ggml_type & val) {
        اكتب(int32_t(val));
    }

    تهی اكتب(ثابت شمارش gguf_type & val) {
        اكتب(int32_t(val));
    }

    تهی اكتب(ثابت ساختار gguf_kv & kv) {
        ثابت uint64_t ne = kv.get_ne();

        اكتب(kv.get_کلید());

        اگر (kv.is_array) {
            اكتب(GGUF_TYPE_ARRAY);
            اكتب(kv.get_type());
            اكتب(ne);
        } وگرنه {
            اكتب(kv.get_type());
        }

        انتخاب (kv.get_type()) {
            حالت GGUF_TYPE_UINT8:
            حالت GGUF_TYPE_INT8:
            حالت GGUF_TYPE_UINT16:
            حالت GGUF_TYPE_INT16:
            حالت GGUF_TYPE_UINT32:
            حالت GGUF_TYPE_INT32:
            حالت GGUF_TYPE_FLOAT32:
            حالت GGUF_TYPE_UINT64:
            حالت GGUF_TYPE_INT64:
            حالت GGUF_TYPE_FLOAT64: {
                اكتب(kv.data);
            } بشکن;
            حالت GGUF_TYPE_BOOL: {
                برای (size_t i = 0; i < ne; ++i) {
                    اكتب(kv.get_val<منطقی>(i));
                }
            } بشکن;
            حالت GGUF_TYPE_STRING: {
                برای (size_t i = 0; i < ne; ++i) {
                    اكتب(kv.get_val<std::string>(i));
                }
            } بشکن;
            حالت GGUF_TYPE_ARRAY:
            پیشفرض: GGML_ABORT("invalid type");
        }
    }

    تهی write_tensor_meta(ثابت ساختار gguf_tensor_info & info) {
        اكتب(info.t.name);

        ثابت uint32_t n_dims = ggml_n_dims(&info.t);
        اكتب(n_dims);

        برای (uint32_t j = 0; j < n_dims; ++j) {
            اكتب(info.t.ne[j]);
        }
        اكتب(info.t.type);
        اكتب(info.offset);
    }

    تهی pad(ثابت size_t alignment) {
        وقتی (written_bytes % alignment != 0) {
            ثابت int8_t zero = 0;
            اكتب(zero);
        }
    }
};

// vector buffer based writer
ساختار gguf_writer_buf final : عمومی gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::اكتب;

    تهی اكتب(ثابت int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    تهی اكتب(ثابت std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    تهی write_tensor_data(ثابت ساختار gguf_tensor_info & info, ثابت size_t offset_data, ثابت size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t offset = buf.size();
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        اگر (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } وگرنه {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
ساختار gguf_writer_file final : عمومی gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::اكتب;

    تهی اكتب(ثابت int8_t val) override {
        ثابت auto real_val = static_cast<uint8_t>(val);
        ثابت auto ret = fputc(real_val, file);
        written_bytes++;
        اگر (ret != real_val) {
            پرتابکن std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((صحیح)real_val) + "'");
        }
    }

    تهی اكتب(ثابت std::vector<int8_t> & val) override {
        ثابت auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        اگر (ret != val.size()) {
            پرتابکن std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    تهی write_tensor_data(ثابت ساختار gguf_tensor_info & info, ثابت size_t offset_data, ثابت size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        اگر (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } وگرنه {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        اكتب(buf);

        pad(alignment);
    }
};

template <typename writer_t>
ایستا تهی gguf_write_out(ثابت ساختار gguf_context * ctx, writer_t & gw, منطقی only_meta) {
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
    برای (int64_t i = 0; i < n_kv; ++i) {
        gw.اكتب(ctx->kv[i]);
    }

    // write tensor info
    برای (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    اگر (only_meta) {
        بازگشت;
    }

    ثابت size_t offset_data = gw.written_bytes;

    // write tensor data
    برای (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

تهی gguf_write_to_buf(ثابت ساختار gguf_context * ctx, std::vector<int8_t> & buf, منطقی only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

منطقی gguf_write_to_file_ptr(ثابت ساختار gguf_context * ctx, FILE * file, منطقی only_meta) {
    GGML_ASSERT(file);

    تلاش {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } بگیر (ثابت std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        بازگشت نادرست;
    }
    بازگشت درست;
}

منطقی gguf_write_to_file(ثابت ساختار gguf_context * ctx, ثابت نویسه * fname, منطقی only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    اگر (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        بازگشت نادرست;
    }

    ثابت منطقی success = gguf_write_to_file_ptr(ctx, file, only_meta);
    اگر (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    بازگشت success;
}

size_t gguf_get_meta_size(ثابت ساختار gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ درست);
    بازگشت buf.size();
}

تهی gguf_get_meta_data(ثابت ساختار gguf_context * ctx, تهی * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ درست);
    memcpy(data, buf.data(), buf.size());
}
