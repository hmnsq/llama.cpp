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

#تعریف_کرو GGUF_MAX_STRING_لمبائی  (1024*1024*1024)
#تعریف_کرو GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#اگر_تعریف_شدہ _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#ورنہ
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#اگر_ختم

سانچہ <قسم_نام T>
ساخت type_مقصد_gguf_type;

سانچہ <>
ساخت type_مقصد_gguf_type<uint8_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_UINT8;
};

سانچہ <>
ساخت type_مقصد_gguf_type<int8_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_INT8;
};

سانچہ <>
ساخت type_مقصد_gguf_type<uint16_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_UINT16;
};

سانچہ <>
ساخت type_مقصد_gguf_type<int16_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_INT16;
};

سانچہ <>
ساخت type_مقصد_gguf_type<uint32_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_UINT32;
};

سانچہ <>
ساخت type_مقصد_gguf_type<int32_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_INT32;
};

سانچہ <>
ساخت type_مقصد_gguf_type<اعشاری> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_FLOAT32;
};

سانچہ <>
ساخت type_مقصد_gguf_type<منطقی> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_BOOL;
};

سانچہ <>
ساخت type_مقصد_gguf_type<std::string> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_STRING;
};

سانچہ <>
ساخت type_مقصد_gguf_type<uint64_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_UINT64;
};

سانچہ <>
ساخت type_مقصد_gguf_type<int64_t> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_INT64;
};

سانچہ <>
ساخت type_مقصد_gguf_type<دوہرا> {
    ساکن constexpr شمار gguf_type value = GGUF_TYPE_FLOAT64;
};

ساکن ثابت std::map<gguf_type, size_t> GGUF_TYPE_سائز = {
    {GGUF_TYPE_UINT8,   حجم(uint8_t)},
    {GGUF_TYPE_INT8,    حجم(int8_t)},
    {GGUF_TYPE_UINT16,  حجم(uint16_t)},
    {GGUF_TYPE_INT16,   حجم(int16_t)},
    {GGUF_TYPE_UINT32,  حجم(uint32_t)},
    {GGUF_TYPE_INT32,   حجم(int32_t)},
    {GGUF_TYPE_FLOAT32, حجم(اعشاری)},
    {GGUF_TYPE_BOOL,    حجم(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  حجم(uint64_t)},
    {GGUF_TYPE_INT64,   حجم(int64_t)},
    {GGUF_TYPE_FLOAT64, حجم(دوہرا)},
};
static_assert(GGUF_TYPE_گنتی == 13, "GGUF_TYPE_COUNT != 13");

ساکن ثابت std::map<gguf_type, ثابت حرف *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_گنتی == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_سائز(شمار gguf_type type) {
    خودکار it = GGUF_TYPE_سائز.find(type);
    واپس it == GGUF_TYPE_سائز.end() ? 0 : it->second;
}

ساخت gguf_kv {
    std::string کلید;

    منطقی is_array;
    شمار gguf_type type;

    std::vector<int8_t>      ڈیٹا;
    std::vector<std::string> ڈیٹا_string;

    سانچہ <قسم_نام T>
    gguf_kv(ثابت std::string & کلید, ثابت T value)
            : کلید(کلید), is_array(جھوٹ), type(type_مقصد_gguf_type<T>::value) {
        GGML_ASSERT(!کلید.خالی_خانے());
        ڈیٹا.resize(حجم(T));
        حافظہ_نقل(ڈیٹا.ڈیٹا(), &value, حجم(T));
    }

    سانچہ <قسم_نام T>
    gguf_kv(ثابت std::string & کلید, ثابت std::vector<T> & value)
            : کلید(کلید), is_array(سچ), type(type_مقصد_gguf_type<T>::value) {
        GGML_ASSERT(!کلید.خالی_خانے());
        ڈیٹا.resize(value.size()*حجم(T));
        برائے (size_t i = 0; i < value.size(); ++i) {
            ثابت T tmp = value[i];
            حافظہ_نقل(ڈیٹا.ڈیٹا() + i*حجم(T), &tmp, حجم(T));
        }
    }

    gguf_kv(ثابت std::string & کلید, ثابت std::string & value)
            : کلید(کلید), is_array(جھوٹ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلید.خالی_خانے());
        ڈیٹا_string.push_back(value);
    }

    gguf_kv(ثابت std::string & کلید, ثابت std::vector<std::string> & value)
            : کلید(کلید), is_array(سچ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلید.خالی_خانے());
        ڈیٹا_string = value;
    }

    ثابت std::string & get_کلید() ثابت {
        واپس کلید;
    }

    ثابت شمار gguf_type & get_type() ثابت {
        واپس type;
    }

    size_t get_ne() ثابت {
        اگر (type == GGUF_TYPE_STRING) {
            ثابت size_t ne = ڈیٹا_string.size();
            GGML_ASSERT(is_array || ne == 1);
            واپس ne;
        }
        ثابت size_t type_سائز = gguf_type_سائز(type);
        GGML_ASSERT(ڈیٹا.size() % type_سائز == 0);
        ثابت size_t ne = ڈیٹا.size() / type_سائز;
        GGML_ASSERT(is_array || ne == 1);
        واپس ne;
    }

    سانچہ <قسم_نام T>
    ثابت T & get_val(ثابت size_t i = 0) ثابت {
        GGML_ASSERT(type_مقصد_gguf_type<T>::value == type);
        اگر constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(ڈیٹا_string.size() >= i+1);
            واپس ڈیٹا_string[i];
        }
        ثابت size_t type_سائز = gguf_type_سائز(type);
        GGML_ASSERT(ڈیٹا.size() % type_سائز == 0);
        GGML_ASSERT(ڈیٹا.size() >= (i+1)*type_سائز);
        واپس reinterpret_cast<ثابت T *>(ڈیٹا.ڈیٹا())[i];
    }

    خالی cast(ثابت شمار gguf_type new_type) {
        ثابت size_t new_type_سائز = gguf_type_سائز(new_type);
        GGML_ASSERT(ڈیٹا.size() % new_type_سائز == 0);
        type = new_type;
    }
};

ساخت gguf_tensor_info {
    ساخت ggml_tensor t; // for holding the equivalent info
    uint64_t انحراف;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

ساخت gguf_context {
    uint32_t نسخہ_2 = GGUF_نسخہ;

    std::vector<ساخت gguf_kv> kv;
    std::vector<ساخت gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t انحراف    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    خالی * ڈیٹا = خالی_اشاریہ;
};

ساخت gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            خالی * userdata,
            size_t max_chunk_read,
            uint64_t ڈیٹا_انحراف = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          آغاز_انحراف(ڈیٹا_انحراف),
          ڈیٹا_انحراف(ڈیٹا_انحراف),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    ساکن uint64_t file_remain(FILE * file) {
        ثابت int64_t cur = gguf_ftell(file);
        اگر (cur < 0) {
            واپس 0;
        }
        اگر (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            واپس 0;
        }
        ثابت int64_t end = gguf_ftell(file);
        اگر (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            واپس 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        واپس static_cast<uint64_t>(end - cur);
    }

    سانچہ <قسم_نام T>
    منطقی اقرأ(T & dst) ثابت {
        ثابت size_t size = حجم(dst);
        اگر (size > nbytes_remain) {
            واپس جھوٹ;
        }
        واپس read_raw(&dst, size) == size;
    }

    سانچہ <قسم_نام T>
    منطقی اقرأ(std::vector<T> & dst, ثابت size_t n) ثابت {
        اگر (n > GGUF_MAX_ARRAY_ELEMENTS) {
            واپس جھوٹ;
        }
        اگر constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            اگر (n > SIZE_MAX / حجم(uint64_t)) {
                واپس جھوٹ;
            }
            اگر (nbytes_remain < n * حجم(uint64_t)) {
                واپس جھوٹ;
            }
        } ورنہ {
            اگر (n > SIZE_MAX / حجم(T)) {
                واپس جھوٹ;
            }
            اگر (nbytes_remain < n * حجم(T)) {
                واپس جھوٹ;
            }
        }
        dst.resize(n);
        برائے (size_t i = 0; i < dst.size(); ++i) {
            اگر constexpr (std::is_same<T, منطقی>::value) {
                منطقی tmp;
                اگر (!اقرأ(tmp)) {
                    واپس جھوٹ;
                }
                dst[i] = tmp;
            } ورنہ {
                اگر (!اقرأ(dst[i])) {
                    واپس جھوٹ;
                }
            }
        }
        واپس سچ;
    }

    منطقی اقرأ(منطقی & dst) ثابت {
        int8_t tmp = -1;
        اگر (!اقرأ(tmp)) {
            واپس جھوٹ;
        }
        dst = tmp != 0;
        واپس سچ;
    }

    منطقی اقرأ(شمار ggml_type & dst) ثابت {
        int32_t tmp = -1;
        اگر (!اقرأ(tmp)) {
            واپس جھوٹ;
        }
        dst = ggml_type(tmp);
        واپس سچ;
    }

    منطقی اقرأ(شمار gguf_type & dst) ثابت {
        int32_t tmp = -1;
        اگر (!اقرأ(tmp)) {
            واپس جھوٹ;
        }
        dst = gguf_type(tmp);
        واپس سچ;
    }

    منطقی اقرأ(std::string & dst) ثابت {
        uint64_t size = 0;
        اگر (!اقرأ(size)) {
            واپس جھوٹ;
        }
        اگر (size > GGUF_MAX_STRING_لمبائی) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_لمبائی);
            واپس جھوٹ;
        }
        اگر (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            واپس جھوٹ;
        }
        dst.resize(static_cast<size_t>(size));
        واپس read_raw(dst.ڈیٹا(), static_cast<size_t>(size)) == size;
    }

    منطقی اقرأ(خالی * dst, ثابت size_t size) ثابت {
        اگر (size > nbytes_remain) {
            واپس جھوٹ;
        }
        واپس read_raw(dst, size) == size;
    }

    uint64_t tell() ثابت {
        واپس ڈیٹا_انحراف;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ثابت {
        واپس آغاز_انحراف;
    }

    منطقی seek(uint64_t absolute_انحراف) ثابت {
        ثابت uint64_t انتہا_انحراف = uint64_t(ڈیٹا_انحراف) + nbytes_remain;
        اگر (absolute_انحراف > انتہا_انحراف) {
            واپس جھوٹ;
        }

        ڈیٹا_انحراف = absolute_انحراف;
        nbytes_remain = انتہا_انحراف - absolute_انحراف;

        واپس سچ;
    }

نجی:
    size_t read_raw(خالی * dst, size_t size) ثابت {
        اگر (callback == خالی_اشاریہ || size == 0) {
            واپس 0;
        }

        uint8_t * ڈیٹا = static_cast<uint8_t *>(dst);
        size_t کل_nread = 0;
        منطقی reached_eof = جھوٹ;

        جب (کل_nread < size) {
            ثابت size_t chunk_سائز = std::min(max_chunk_read, size - کل_nread);
            اگر (ڈیٹا_انحراف + کل_nread < ڈیٹا_انحراف) {
                توڑو;
            }
            ثابت size_t nread = callback(userdata, static_cast<خالی *>(ڈیٹا + کل_nread), ڈیٹا_انحراف + کل_nread, chunk_سائز);
            کل_nread += nread;
            اگر (nread != chunk_سائز) {
                reached_eof = سچ;
                توڑو;
            }
        }

        ڈیٹا_انحراف += کل_nread;
        GGML_ASSERT(کل_nread <= nbytes_remain);
        nbytes_remain -= کل_nread;

        اگر (reached_eof) {
            nbytes_remain = 0;
        }

        واپس کل_nread;
    }

    gguf_reader_callback_t callback = خالی_اشاریہ;
    خالی * userdata = خالی_اشاریہ;
    size_t max_chunk_read = 0;
    uint64_t آغاز_انحراف = 0;
    تغیرپذیر uint64_t ڈیٹا_انحراف = 0;
    تغیرپذیر uint64_t nbytes_remain = 0;
};

ساخت gguf_context * gguf_init_خالی_خانے(خالی) {
    واپس نیا gguf_context;
}

سانچہ<قسم_نام T>
منطقی gguf_read_emplace_helper(ثابت ساخت gguf_reader & gr, std::vector<ساخت gguf_kv> & kv, ثابت std::string & کلید, ثابت منطقی is_array, ثابت size_t n) {
    اگر (is_array) {
        std::vector<T> value;
        کوشش {
            اگر (!gr.اقرأ(value, n)) {
                واپس جھوٹ;
            }
        } پکڑو (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, کلید.c_str());
            واپس جھوٹ;
        } پکڑو (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, کلید.c_str());
            واپس جھوٹ;
        }
        kv.emplace_back(کلید, value);
    } ورنہ {
        T value;
        اگر (!gr.اقرأ(value)) {
            واپس جھوٹ;
        }
        kv.emplace_back(کلید, value);
    }
    واپس سچ;
}

ساکن ساخت gguf_context * gguf_init_from_reader(ثابت ساخت gguf_reader & gr, ساخت gguf_init_params params) {
    ساخت gguf_context * ctx = نیا gguf_context;

    منطقی ok = سچ;

    // file magic
    {
        std::vector<حرف> magic;
        ok = ok && gr.اقرأ(magic, 4);

        اگر (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            واپس خالی_اشاریہ;
        }

        برائے (uint32_t i = 0; i < magic.size(); i++) {
            اگر (magic[i] != GGUF_MAGIC[i]) {
                حرف c0 = isprint(magic[0]) ? magic[0] : '?';
                حرف c1 = isprint(magic[1]) ? magic[1] : '?';
                حرف c2 = isprint(magic[2]) ? magic[2] : '?';
                حرف c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }
        }
    }

    //سرخی
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    اگر (ok && gr.اقرأ(ctx->نسخہ_2)) {
        اگر (ok && ctx->نسخہ_2 == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->نسخہ_2);
            ok = جھوٹ;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        اگر (ok && (ctx->نسخہ_2 & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->نسخہ_2);
            ok = جھوٹ;
        }

        اگر (ok && ctx->نسخہ_2 == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = جھوٹ;
        }
        اگر (ok && ctx->نسخہ_2 > GGUF_نسخہ) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->نسخہ_2, GGUF_نسخہ);
            ok = جھوٹ;
        }
    } ورنہ {
        ok = جھوٹ;
    }

    اگر (ok && gr.اقرأ(n_tensors)) {
        static_assert(حجم(size_t) <= 8 && حجم(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        اگر (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/حجم(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/حجم(gguf_tensor_info));
            ok = جھوٹ;
        }
    } ورنہ {
        ok = جھوٹ;
    }

    اگر (ok && gr.اقرأ(n_kv)) {
        static_assert(حجم(size_t) <= 8 && حجم(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        اگر (n_kv < 0 || n_kv > int64_t(SIZE_MAX/حجم(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/حجم(gguf_kv));
            ok = جھوٹ;
        }
    } ورنہ {
        ok = جھوٹ;
    }

    اگر (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        واپس خالی_اشاریہ;
    }

    // KV pairs
    {
        std::unordered_set<std::string> نظر_آیا_کلیدیں;

        برائے (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string کلید;
            gguf_type   type     = gguf_type(-1);
            منطقی        is_array = جھوٹ;
            uint64_t    n        = 1;

            کوشش {
                ok = ok && gr.اقرأ(کلید);
            } پکڑو (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = جھوٹ;
            } پکڑو (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = جھوٹ;
            }
            اگر (ok && کلید.خالی_خانے()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = جھوٹ;
            }
            اگر (ok && !نظر_آیا_کلیدیں.insert(کلید).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, کلید.c_str(), i);
                ok = جھوٹ;
            }
            اگر (!ok) {
                توڑو;
            }

            ok = ok && gr.اقرأ(type);
            اگر (type == GGUF_TYPE_ARRAY) {
                is_array = سچ;
                ok = ok && gr.اقرأ(type);
                ok = ok && gr.اقرأ(n);
            }
            اگر (!ok) {
                توڑو;
            }

            انتخاب (type) {
                حالت GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<اعشاری>      (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<منطقی>       (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<دوہرا>     (gr, ctx->kv, کلید, is_array, n); توڑو;
                حالت GGUF_TYPE_ARRAY:
                طےشدہ:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, کلید.c_str(), type);
                        ok = جھوٹ;
                    } توڑو;
            }
        }

        اگر (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            واپس خالی_اشاریہ;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ثابت صحیح alignment_idx = gguf_ڈھونڈو_کلید(ctx, GGUF_کلید_GENERAL_ALIGNMENT);
        اگر (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_کلید_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            واپس خالی_اشاریہ;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        اگر (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            واپس خالی_اشاریہ;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> نظر_آیا_tensor_نام;

    برائے (int64_t i = 0; ok && i < n_tensors; ++i) {
        ساخت gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            کوشش {
                ok = ok && gr.اقرأ(name);
            } پکڑو (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = جھوٹ;
            } پکڑو (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = جھوٹ;
            }
            اگر (name.لمبائی() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.لمبائی(), GGML_MAX_NAME);
                ok = جھوٹ;
                توڑو;
            }
            ggml_سیٹ_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            اگر (ok && !نظر_آیا_tensor_نام.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = جھوٹ;
                توڑو;
            }
        }
        اگر (!ok) {
            توڑو;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.اقرأ(n_dims);
            اگر (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = جھوٹ;
                توڑو;
            }
            برائے (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                اگر (j < n_dims) {
                    ok = ok && gr.اقرأ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                اگر (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = جھوٹ;
                    توڑو;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            اگر (ok) {
                int64_t current_nelements = info.t.ne[0];
                برائے (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
                    اگر (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        اگر (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = جھوٹ;
                             توڑو;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 اگر (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     توڑو;
                 }
            }
        }
        اگر (!ok) {
            توڑو;
        }

        // tensor type
        {
            ok = ok && gr.اقرأ(info.t.type);

            // check that tensor type is within defined range
            اگر (info.t.type < 0 || info.t.type >= GGML_TYPE_گنتی) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_گنتی);
                ok = جھوٹ;
                توڑو;
            }
            ثابت size_t  type_سائز = ggml_type_سائز(info.t.type);
            ثابت int64_t blck_سائز = ggml_blck_سائز(info.t.type);

            // check that row size is divisible by block size
            اگر (blck_سائز == 0 || info.t.ne[0] % blck_سائز != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (صحیح) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_سائز);
                ok = جھوٹ;
                توڑو;
            }

            // check that the size of the tensor in bytes is representable
            اگر (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_سائز(info.t.type)) > SIZE_MAX/ggml_type_سائز(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = جھوٹ;
                توڑو;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_سائز;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_سائز);
            برائے (صحیح j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        اگر (!ok) {
            توڑو;
        }

        // tensor data offset within buffer
        ok = ok && gr.اقرأ(info.انحراف);

        ctx->info.push_back(info);
    }

    اگر (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        واپس خالی_اشاریہ;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    اگر (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        واپس خالی_اشاریہ;
    }

    // store the current file offset - this is where the data section starts
    ctx->انحراف = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        برائے (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت gguf_tensor_info & ti = ctx->info[i];
            اگر (ti.انحراف != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.انحراف, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }
            ثابت size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            اگر (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }
            ثابت size_t padded_سائز = GGML_PAD(nbytes, ctx->alignment);
            اگر (SIZE_MAX - ctx->size < padded_سائز) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_سائز);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }
            ctx->size += padded_سائز;
        }
    }

    // load the tensor data only if requested
    اگر (params.ctx != خالی_اشاریہ) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_سائز = 0;
        اگر (params.no_alloc) {
            اگر (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }

            ثابت size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_سائز = overhead;
        } ورنہ {
            اگر ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }

            ثابت size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            اگر (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }

            mem_سائز = overhead + ctx->size;
        }

        ساخت ggml_init_params pdata = {
            /*mem_size   =*/ mem_سائز,
            /*mem_buffer =*/ خالی_اشاریہ,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        اگر (*params.ctx == خالی_اشاریہ) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            واپس خالی_اشاریہ;
        }

        ساخت ggml_context * ctx_ڈیٹا = *params.ctx;

        ساخت ggml_tensor * ڈیٹا = خالی_اشاریہ;

        اگر (!params.no_alloc) {
            ڈیٹا = ggml_new_tensor_1d(ctx_ڈیٹا, GGML_TYPE_I8, ctx->size);

            ok = ok && ڈیٹا != خالی_اشاریہ;

            اگر (ok) {
                ggml_سیٹ_name(ڈیٹا, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.اقرأ(ڈیٹا->ڈیٹا, ctx->size);

            اگر (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_ڈیٹا);
                *params.ctx = خالی_اشاریہ;
                gguf_free(ctx);
                واپس خالی_اشاریہ;
            }

            ctx->ڈیٹا = ڈیٹا->ڈیٹا;
        }

        ggml_سیٹ_no_alloc(ctx_ڈیٹا, سچ);

        // create the tensors
        برائے (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت ساخت gguf_tensor_info & info = ctx->info[i];

            ساخت ggml_tensor * cur = ggml_new_tensor(ctx_ڈیٹا, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != خالی_اشاریہ;

            اگر (!ok) {
                توڑو;
            }

            ggml_سیٹ_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            اگر (!params.no_alloc) {
                cur->ڈیٹا = (حرف *) ڈیٹا->ڈیٹا + info.انحراف;
            }
        }

        اگر (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_ڈیٹا);
            *params.ctx = خالی_اشاریہ;
            gguf_free(ctx);
            واپس خالی_اشاریہ;
        }

        ggml_سیٹ_no_alloc(ctx_ڈیٹا, params.no_alloc);
    }

    واپس ctx;
}

ساخت gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, خالی * userdata, size_t max_chunk_read, uint64_t max_expected_سائز, ساخت gguf_init_params params) {
    اگر (callback == خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }

    ثابت ساخت gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_سائز);
    واپس gguf_init_from_reader(gr, params);
}

ساخت gguf_file_reader {
    FILE * file;
    uint64_t انحراف;
};

ساکن size_t gguf_file_reader_callback(خالی * userdata, خالی * output, uint64_t انحراف, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    اگر (reader.انحراف != انحراف) {
        اگر (انحراف > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(انحراف), SEEK_SET) != 0) {
            واپس 0;
        }

        reader.انحراف = انحراف;
    }

    ثابت size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.انحراف += nread;
    واپس nread;
}

ساخت gguf_context * gguf_init_from_file_ptr(FILE * file, ساخت gguf_init_params params) {
    اگر (!file) {
        واپس خالی_اشاریہ;
    }

    ثابت int64_t cur = gguf_ftell(file);
    اگر (cur < 0) {
        واپس خالی_اشاریہ;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ثابت ساخت gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.انحراف, gguf_reader::file_remain(file));
    واپس gguf_init_from_reader(gr, params);
}

ساخت gguf_بفر_reader {
    ثابت uint8_t * ڈیٹا;
    size_t          size;
};

ساکن size_t gguf_بفر_reader_callback(خالی * userdata, خالی * output, uint64_t انحراف, size_t len) {
    GGML_ASSERT(len > 0);

    ثابت gguf_بفر_reader & reader = *static_cast<gguf_بفر_reader *>(userdata);

    اگر (انحراف > reader.size || len > reader.size - انحراف) {
        واپس 0;
    }

    ثابت size_t ڈیٹا_انحراف = static_cast<size_t>(انحراف);
    ثابت size_t nread = std::min(len, reader.size - ڈیٹا_انحراف);
    حافظہ_نقل(static_cast<uint8_t *>(output), reader.ڈیٹا + ڈیٹا_انحراف, nread);
    واپس nread;
}

ساخت gguf_context * gguf_init_from_بفر(ثابت خالی * ڈیٹا, size_t size, ساخت gguf_init_params params) {
    اگر (ڈیٹا == خالی_اشاریہ || size == 0) {
        واپس خالی_اشاریہ;
    }

    gguf_بفر_reader reader = {
        /*.data = */ static_cast<ثابت uint8_t *>(ڈیٹا),
        /*.size = */ size,
    };
    ثابت ساخت gguf_reader gr(gguf_بفر_reader_callback, &reader, SIZE_MAX, 0, size);
    واپس gguf_init_from_reader(gr, params);
}

ساخت gguf_context * gguf_init_from_file(ثابت حرف * fname, ساخت gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    اگر (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        واپس خالی_اشاریہ;
    }

    ساخت gguf_context * نتیجہ = gguf_init_from_file_ptr(file, params);
    fclose(file);
    واپس نتیجہ;
}

خالی gguf_free(ساخت gguf_context * ctx) {
    اگر (ctx == خالی_اشاریہ) {
        واپس;
    }
    حذف ctx;
}

ثابت حرف * gguf_type_name(شمار gguf_type type) {
    خودکار it = GGUF_TYPE_NAME.find(type);
    واپس it == GGUF_TYPE_NAME.end() ? خالی_اشاریہ : it->second;
}

uint32_t gguf_get_نسخہ(ثابت ساخت gguf_context * ctx) {
    واپس ctx->نسخہ_2;
}

size_t gguf_get_alignment(ثابت ساخت gguf_context * ctx) {
    واپس ctx->alignment;
}

size_t gguf_get_ڈیٹا_انحراف(ثابت ساخت gguf_context * ctx) {
    واپس ctx->انحراف;
}

int64_t gguf_get_n_kv(ثابت ساخت gguf_context * ctx) {
    واپس ctx->kv.size();
}

int64_t gguf_ڈھونڈو_کلید(ثابت ساخت gguf_context * ctx, ثابت حرف * کلید) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ثابت int64_t n_kv = gguf_get_n_kv(ctx);

    برائے (int64_t i = 0; i < n_kv; ++i) {
        اگر (strcmp(کلید, gguf_get_کلید(ctx, i)) == 0) {
            keyfound = i;
            توڑو;
        }
    }

    واپس keyfound;
}

ثابت حرف * gguf_get_کلید(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    واپس ctx->kv[کلید_id].get_کلید().c_str();
}

شمار gguf_type gguf_get_kv_type(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    واپس ctx->kv[کلید_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[کلید_id].get_type();
}

شمار gguf_type gguf_get_arr_type(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].is_array);
    واپس ctx->kv[کلید_id].get_type();
}

ثابت خالی * gguf_get_arr_ڈیٹا(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_type() != GGUF_TYPE_STRING);
    واپس ctx->kv[کلید_id].ڈیٹا.ڈیٹا();
}

ثابت حرف * gguf_get_arr_str(ثابت ساخت gguf_context * ctx, int64_t کلید_id, size_t i) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_type() == GGUF_TYPE_STRING);
    واپس ctx->kv[کلید_id].ڈیٹا_string[i].c_str();
}

size_t gguf_get_arr_n(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));

    اگر (ctx->kv[کلید_id].type == GGUF_TYPE_STRING) {
        واپس ctx->kv[کلید_id].ڈیٹا_string.size();
    }

    ثابت size_t type_سائز = gguf_type_سائز(ctx->kv[کلید_id].type);
    GGML_ASSERT(ctx->kv[کلید_id].ڈیٹا.size() % type_سائز == 0);
    واپس ctx->kv[کلید_id].ڈیٹا.size() / type_سائز;
}

uint8_t gguf_get_val_u8(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<int32_t>();
}

اعشاری gguf_get_val_f32(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<اعشاری>();
}

uint64_t gguf_get_val_u64(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<int64_t>();
}

دوہرا gguf_get_val_f64(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<دوہرا>();
}

منطقی gguf_get_val_bool(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<منطقی>();
}

ثابت حرف * gguf_get_val_str(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    واپس ctx->kv[کلید_id].get_val<std::string>().c_str();
}

ثابت خالی * gguf_get_val_ڈیٹا(ثابت ساخت gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[کلید_id].get_type() != GGUF_TYPE_STRING);
    واپس ctx->kv[کلید_id].ڈیٹا.ڈیٹا();
}

int64_t gguf_get_n_tensors(ثابت ساخت gguf_context * ctx) {
    واپس ctx->info.size();
}

int64_t gguf_ڈھونڈو_tensor(ثابت ساخت gguf_context * ctx, ثابت حرف * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    برائے (int64_t i = 0; i < n_tensors; ++i) {
        اگر (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            توڑو;
        }
    }

    واپس tensor_id;
}

size_t gguf_get_tensor_انحراف(ثابت ساخت gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    واپس ctx->info[tensor_id].انحراف;
}

ثابت حرف * gguf_get_tensor_name(ثابت ساخت gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    واپس ctx->info[tensor_id].t.name;
}

ثابت int64_t * gguf_get_tensor_ne(ثابت ساخت gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    واپس ctx->info[tensor_id].t.ne;
}

شمار ggml_type gguf_get_tensor_type(ثابت ساخت gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    واپس ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_سائز(ثابت ساخت gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    واپس ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_کلید(ساخت gguf_context * ctx, ثابت حرف * کلید) {
    ثابت int64_t کلید_id = gguf_ڈھونڈو_کلید(ctx, کلید);
    اگر (کلید_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + کلید_id);
    }
    واپس کلید_id;
}

سانچہ<قسم_نام T>
ساکن خالی gguf_check_reserved_کلیدیں(ثابت std::string & کلید, ثابت T val) {
    اگر (کلید == GGUF_کلید_GENERAL_ALIGNMENT) {
        اگر constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_کلید_GENERAL_ALIGNMENT " must be power of 2");
        } ورنہ {
            GGML_غیر_مستعمل(val);
            GGML_ABORT(GGUF_کلید_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

خالی gguf_سیٹ_val_u8(ساخت gguf_context * ctx, ثابت حرف * کلید, uint8_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_i8(ساخت gguf_context * ctx, ثابت حرف * کلید, int8_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_u16(ساخت gguf_context * ctx, ثابت حرف * کلید, uint16_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_i16(ساخت gguf_context * ctx, ثابت حرف * کلید, int16_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_u32(ساخت gguf_context * ctx, ثابت حرف * کلید, uint32_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_i32(ساخت gguf_context * ctx, ثابت حرف * کلید, int32_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_f32(ساخت gguf_context * ctx, ثابت حرف * کلید, اعشاری val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_u64(ساخت gguf_context * ctx, ثابت حرف * کلید, uint64_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_i64(ساخت gguf_context * ctx, ثابت حرف * کلید, int64_t val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_f64(ساخت gguf_context * ctx, ثابت حرف * کلید, دوہرا val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_bool(ساخت gguf_context * ctx, ثابت حرف * کلید, منطقی val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

خالی gguf_سیٹ_val_str(ساخت gguf_context * ctx, ثابت حرف * کلید, ثابت حرف * val) {
    gguf_check_reserved_کلیدیں(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, std::string(val));
}

خالی gguf_سیٹ_arr_ڈیٹا(ساخت gguf_context * ctx, ثابت حرف * کلید, شمار gguf_type type, ثابت خالی * ڈیٹا, size_t n) {
    gguf_check_reserved_کلیدیں(کلید, ڈیٹا);
    gguf_remove_کلید(ctx, کلید);

    ثابت size_t nbytes = n*gguf_type_سائز(type);
    std::vector<int8_t> tmp(nbytes);
    اگر (!tmp.خالی_خانے()) {
        حافظہ_نقل(tmp.ڈیٹا(), ڈیٹا, nbytes);
    }
    ctx->kv.emplace_back(کلید, tmp);
    ctx->kv.back().cast(type);
}

خالی gguf_سیٹ_arr_str(ساخت gguf_context * ctx, ثابت حرف * کلید, ثابت حرف ** ڈیٹا, size_t n) {
    gguf_check_reserved_کلیدیں(کلید, ڈیٹا);
    gguf_remove_کلید(ctx, کلید);

    std::vector<std::string> tmp(n);
    برائے (size_t i = 0; i < n; ++i) {
        tmp[i] = ڈیٹا[i];
    }
    ctx->kv.emplace_back(کلید, tmp);
}

// set or add KV pairs from another context
خالی gguf_سیٹ_kv(ساخت gguf_context * ctx, ثابت ساخت gguf_context * src) {
    ثابت int64_t n_kv = gguf_get_n_kv(src);
    برائے (int64_t i = 0; i < n_kv; ++i) {
        ثابت ساخت gguf_kv & kv = src->kv[i];

        اگر (!kv.is_array) {
            انتخاب (kv.get_type()) {
                حالت GGUF_TYPE_UINT8:   gguf_سیٹ_val_u8  (ctx, kv.get_کلید().c_str(), kv.get_val<uint8_t>());             توڑو;
                حالت GGUF_TYPE_INT8:    gguf_سیٹ_val_i8  (ctx, kv.get_کلید().c_str(), kv.get_val<int8_t>());              توڑو;
                حالت GGUF_TYPE_UINT16:  gguf_سیٹ_val_u16 (ctx, kv.get_کلید().c_str(), kv.get_val<uint16_t>());            توڑو;
                حالت GGUF_TYPE_INT16:   gguf_سیٹ_val_i16 (ctx, kv.get_کلید().c_str(), kv.get_val<int16_t>());             توڑو;
                حالت GGUF_TYPE_UINT32:  gguf_سیٹ_val_u32 (ctx, kv.get_کلید().c_str(), kv.get_val<uint32_t>());            توڑو;
                حالت GGUF_TYPE_INT32:   gguf_سیٹ_val_i32 (ctx, kv.get_کلید().c_str(), kv.get_val<int32_t>());             توڑو;
                حالت GGUF_TYPE_FLOAT32: gguf_سیٹ_val_f32 (ctx, kv.get_کلید().c_str(), kv.get_val<اعشاری>());               توڑو;
                حالت GGUF_TYPE_UINT64:  gguf_سیٹ_val_u64 (ctx, kv.get_کلید().c_str(), kv.get_val<uint64_t>());            توڑو;
                حالت GGUF_TYPE_INT64:   gguf_سیٹ_val_i64 (ctx, kv.get_کلید().c_str(), kv.get_val<int64_t>());             توڑو;
                حالت GGUF_TYPE_FLOAT64: gguf_سیٹ_val_f64 (ctx, kv.get_کلید().c_str(), kv.get_val<دوہرا>());              توڑو;
                حالت GGUF_TYPE_BOOL:    gguf_سیٹ_val_bool(ctx, kv.get_کلید().c_str(), kv.get_val<منطقی>());                توڑو;
                حالت GGUF_TYPE_STRING:  gguf_سیٹ_val_str (ctx, kv.get_کلید().c_str(), kv.get_val<std::string>().c_str()); توڑو;
                حالت GGUF_TYPE_ARRAY:
                طےشدہ: GGML_ABORT("invalid type");
            }
            جاری;
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
                gguf_سیٹ_arr_ڈیٹا(ctx, kv.get_کلید().c_str(), kv.get_type(), kv.ڈیٹا.ڈیٹا(), ne);
            } توڑو;
            حالت GGUF_TYPE_STRING: {
                std::vector<ثابت حرف *> tmp(ne);
                برائے (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.ڈیٹا_string[j].c_str();
                }
                gguf_سیٹ_arr_str(ctx, kv.get_کلید().c_str(), tmp.ڈیٹا(), ne);
            } توڑو;
            حالت GGUF_TYPE_ARRAY:
            طےشدہ: GGML_ABORT("invalid type");
        }
    }
}

خالی gguf_جوڑو_tensor(
             ساخت gguf_context * ctx,
        ثابت ساخت ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    اگر (gguf_ڈھونڈو_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    ساخت gguf_tensor_info ti;
    ti.t = *tensor;
    ti.انحراف = ctx->info.خالی_خانے() ? 0 :
        ctx->info.back().انحراف + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

خالی gguf_سیٹ_tensor_type(ساخت gguf_context * ctx, ثابت حرف * name, شمار ggml_type type) {
    ثابت int64_t tensor_id = gguf_ڈھونڈو_tensor(ctx, name);
    اگر (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    ساخت ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ثابت size_t  type_سائز = ggml_type_سائز(type);
    ثابت int64_t blck_سائز = ggml_blck_سائز(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_سائز == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_سائز;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_سائز);
    برائے (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);
    برائے (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].انحراف = ctx->info[i - 1].انحراف + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

خالی gguf_سیٹ_tensor_ڈیٹا(ساخت gguf_context * ctx, ثابت حرف * name, ثابت خالی * ڈیٹا) {
    ثابت int64_t tensor_id = gguf_ڈھونڈو_tensor(ctx, name);
    اگر (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.ڈیٹا = (خالی *)(uintptr_t)ڈیٹا; // double cast suppresses warning about casting away const
}

ساخت gguf_writer_بنیاد {
    size_t written_بائٹیں {0u};

    مجازی ~gguf_writer_بنیاد(خالی) = طےشدہ;

    // we bet on devirtualization
    مجازی خالی اكتب(int8_t val) = 0;
    مجازی خالی اكتب(ثابت std::vector<int8_t> & val) = 0;
    مجازی خالی write_tensor_ڈیٹا(ثابت ساخت gguf_tensor_info & info, size_t انحراف_ڈیٹا, size_t alignment) = 0;

    سانچہ <قسم_نام T>
    خالی اكتب(ثابت T & val) {
        برائے (size_t i = 0; i < حجم(val); ++i) {
            اكتب(reinterpret_cast<ثابت int8_t *>(&val)[i]);
        }
    }

    خالی اكتب(ثابت منطقی & val) {
        ثابت int8_t val8 = val ? 1 : 0;
        اكتب(val8);
    }

    خالی اكتب(ثابت std::string & val) {
        {
            ثابت uint64_t n = val.لمبائی();
            اكتب(n);
        }
        برائے (size_t i = 0; i < val.لمبائی(); ++i) {
            اكتب((val.ڈیٹا())[i]);
        }
    }

    خالی اكتب(ثابت حرف * val) {
        اكتب(std::string(val));
    }

    خالی اكتب(ثابت شمار ggml_type & val) {
        اكتب(int32_t(val));
    }

    خالی اكتب(ثابت شمار gguf_type & val) {
        اكتب(int32_t(val));
    }

    خالی اكتب(ثابت ساخت gguf_kv & kv) {
        ثابت uint64_t ne = kv.get_ne();

        اكتب(kv.get_کلید());

        اگر (kv.is_array) {
            اكتب(GGUF_TYPE_ARRAY);
            اكتب(kv.get_type());
            اكتب(ne);
        } ورنہ {
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
                اكتب(kv.ڈیٹا);
            } توڑو;
            حالت GGUF_TYPE_BOOL: {
                برائے (size_t i = 0; i < ne; ++i) {
                    اكتب(kv.get_val<منطقی>(i));
                }
            } توڑو;
            حالت GGUF_TYPE_STRING: {
                برائے (size_t i = 0; i < ne; ++i) {
                    اكتب(kv.get_val<std::string>(i));
                }
            } توڑو;
            حالت GGUF_TYPE_ARRAY:
            طےشدہ: GGML_ABORT("invalid type");
        }
    }

    خالی write_tensor_meta(ثابت ساخت gguf_tensor_info & info) {
        اكتب(info.t.name);

        ثابت uint32_t n_dims = ggml_n_dims(&info.t);
        اكتب(n_dims);

        برائے (uint32_t j = 0; j < n_dims; ++j) {
            اكتب(info.t.ne[j]);
        }
        اكتب(info.t.type);
        اكتب(info.انحراف);
    }

    خالی pad(ثابت size_t alignment) {
        جب (written_بائٹیں % alignment != 0) {
            ثابت int8_t zero = 0;
            اكتب(zero);
        }
    }
};

// vector buffer based writer
ساخت gguf_writer_buf final : عوامی gguf_writer_بنیاد {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    استعمال gguf_writer_بنیاد::اكتب;

    خالی اكتب(ثابت int8_t val) override {
        buf.push_back(val);
        written_بائٹیں++;
    }

    خالی اكتب(ثابت std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_بائٹیں += val.size();
    }

    خالی write_tensor_ڈیٹا(ثابت ساخت gguf_tensor_info & info, ثابت size_t انحراف_ڈیٹا, ثابت size_t alignment) override {
        GGML_ASSERT(buf.size() - انحراف_ڈیٹا == info.انحراف);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t انحراف = buf.size();
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(انحراف + nbytes);
        اگر (info.t.بفر) {
            ggml_backend_tensor_get(&info.t, buf.ڈیٹا() + انحراف, 0, nbytes);
        } ورنہ {
            GGML_ASSERT(info.t.ڈیٹا);
            حافظہ_نقل(buf.ڈیٹا() + انحراف, info.t.ڈیٹا, nbytes);
        }
        written_بائٹیں += nbytes;

        pad(alignment);
    }
};

// file based writer
ساخت gguf_writer_file final : عوامی gguf_writer_بنیاد {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    استعمال gguf_writer_بنیاد::اكتب;

    خالی اكتب(ثابت int8_t val) override {
        ثابت خودکار real_val = static_cast<uint8_t>(val);
        ثابت خودکار ret = fputc(real_val, file);
        written_بائٹیں++;
        اگر (ret != real_val) {
            پھینکو std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((صحیح)real_val) + "'");
        }
    }

    خالی اكتب(ثابت std::vector<int8_t> & val) override {
        ثابت خودکار ret = fwrite(val.ڈیٹا(), 1, val.size(), file);
        written_بائٹیں += val.size();
        اگر (ret != val.size()) {
            پھینکو std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    خالی write_tensor_ڈیٹا(ثابت ساخت gguf_tensor_info & info, ثابت size_t انحراف_ڈیٹا, ثابت size_t alignment) override {
        GGML_ASSERT(written_بائٹیں - انحراف_ڈیٹا == info.انحراف);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        اگر (info.t.بفر) {
            ggml_backend_tensor_get(&info.t, buf.ڈیٹا(), 0, nbytes);
        } ورنہ {
            GGML_ASSERT(info.t.ڈیٹا);
            حافظہ_نقل(buf.ڈیٹا(), info.t.ڈیٹا, nbytes);
        }
        اكتب(buf);

        pad(alignment);
    }
};

سانچہ <قسم_نام writer_t>
ساکن خالی gguf_write_برآمد(ثابت ساخت gguf_context * ctx, writer_t & gw, منطقی only_meta) {
    ثابت int64_t n_kv      = gguf_get_n_kv(ctx);
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.اكتب(GGUF_MAGIC[0]);
    gw.اكتب(GGUF_MAGIC[1]);
    gw.اكتب(GGUF_MAGIC[2]);
    gw.اكتب(GGUF_MAGIC[3]);
    gw.اكتب(ctx->نسخہ_2);
    gw.اكتب(n_tensors);
    gw.اكتب(n_kv);

    // write key-value pairs
    برائے (int64_t i = 0; i < n_kv; ++i) {
        gw.اكتب(ctx->kv[i]);
    }

    // write tensor info
    برائے (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    اگر (only_meta) {
        واپس;
    }

    ثابت size_t انحراف_ڈیٹا = gw.written_بائٹیں;

    // write tensor data
    برائے (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_ڈیٹا(ctx->info[i], انحراف_ڈیٹا, ctx->alignment);
    }
}

خالی gguf_write_مقصد_buf(ثابت ساخت gguf_context * ctx, std::vector<int8_t> & buf, منطقی only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_برآمد(ctx, gw, only_meta);
}

منطقی gguf_write_مقصد_file_ptr(ثابت ساخت gguf_context * ctx, FILE * file, منطقی only_meta) {
    GGML_ASSERT(file);

    کوشش {
        gguf_writer_file gw(file);
        gguf_write_برآمد(ctx, gw, only_meta);
    } پکڑو (ثابت std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        واپس جھوٹ;
    }
    واپس سچ;
}

منطقی gguf_write_مقصد_file(ثابت ساخت gguf_context * ctx, ثابت حرف * fname, منطقی only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    اگر (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        واپس جھوٹ;
    }

    ثابت منطقی success = gguf_write_مقصد_file_ptr(ctx, file, only_meta);
    اگر (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    واپس success;
}

size_t gguf_get_meta_سائز(ثابت ساخت gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_مقصد_buf(ctx, buf, /*only_meta =*/ سچ);
    واپس buf.size();
}

خالی gguf_get_meta_ڈیٹا(ثابت ساخت gguf_context * ctx, خالی * ڈیٹا) {
    std::vector<int8_t> buf;
    gguf_write_مقصد_buf(ctx, buf, /*only_meta =*/ سچ);
    حافظہ_نقل(ڈیٹا, buf.ڈیٹا(), buf.size());
}
