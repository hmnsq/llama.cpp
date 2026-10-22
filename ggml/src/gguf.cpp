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

#تعریف_کن GGUF_MAX_STRING_درازی  (1024*1024*1024)
#تعریف_کن GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#اگر_تعریف_شده _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#وإلا
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#پایان_اگر

قالب <نام_نوع T>
بنية type_مقصد_gguf_type;

قالب <>
بنية type_مقصد_gguf_type<uint8_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT8;
};

قالب <>
بنية type_مقصد_gguf_type<int8_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT8;
};

قالب <>
بنية type_مقصد_gguf_type<uint16_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT16;
};

قالب <>
بنية type_مقصد_gguf_type<int16_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT16;
};

قالب <>
بنية type_مقصد_gguf_type<uint32_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT32;
};

قالب <>
بنية type_مقصد_gguf_type<int32_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT32;
};

قالب <>
بنية type_مقصد_gguf_type<عائم> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_FLOAT32;
};

قالب <>
بنية type_مقصد_gguf_type<منطقي> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_BOOL;
};

قالب <>
بنية type_مقصد_gguf_type<std::string> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_STRING;
};

قالب <>
بنية type_مقصد_gguf_type<uint64_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT64;
};

قالب <>
بنية type_مقصد_gguf_type<int64_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT64;
};

قالب <>
بنية type_مقصد_gguf_type<مزدوج> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_FLOAT64;
};

ساكن ثابت std::map<gguf_type, size_t> GGUF_TYPE_اندازه = {
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
static_assert(GGUF_TYPE_شمار == 13, "GGUF_TYPE_COUNT != 13");

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
static_assert(GGUF_TYPE_شمار == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_اندازه(تعداد gguf_type type) {
    خودکار it = GGUF_TYPE_اندازه.find(type);
    أرجع it == GGUF_TYPE_اندازه.end() ? 0 : it->second;
}

بنية gguf_kv {
    std::string کلید;

    منطقي is_array;
    تعداد gguf_type type;

    std::vector<int8_t>      داده;
    std::vector<std::string> داده_string;

    قالب <نام_نوع T>
    gguf_kv(ثابت std::string & کلید, ثابت T value)
            : کلید(کلید), is_array(خطأ), type(type_مقصد_gguf_type<T>::value) {
        GGML_ASSERT(!کلید.خالی());
        داده.resize(حجم(T));
        کپی_حافظه(داده.داده(), &value, حجم(T));
    }

    قالب <نام_نوع T>
    gguf_kv(ثابت std::string & کلید, ثابت std::vector<T> & value)
            : کلید(کلید), is_array(درست), type(type_مقصد_gguf_type<T>::value) {
        GGML_ASSERT(!کلید.خالی());
        داده.resize(value.size()*حجم(T));
        لكل (size_t i = 0; i < value.size(); ++i) {
            ثابت T tmp = value[i];
            کپی_حافظه(داده.داده() + i*حجم(T), &tmp, حجم(T));
        }
    }

    gguf_kv(ثابت std::string & کلید, ثابت std::string & value)
            : کلید(کلید), is_array(خطأ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلید.خالی());
        داده_string.push_back(value);
    }

    gguf_kv(ثابت std::string & کلید, ثابت std::vector<std::string> & value)
            : کلید(کلید), is_array(درست), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلید.خالی());
        داده_string = value;
    }

    ثابت std::string & get_کلید() ثابت {
        أرجع کلید;
    }

    ثابت تعداد gguf_type & get_type() ثابت {
        أرجع type;
    }

    size_t get_ne() ثابت {
        إذا (type == GGUF_TYPE_STRING) {
            ثابت size_t ne = داده_string.size();
            GGML_ASSERT(is_array || ne == 1);
            أرجع ne;
        }
        ثابت size_t type_اندازه = gguf_type_اندازه(type);
        GGML_ASSERT(داده.size() % type_اندازه == 0);
        ثابت size_t ne = داده.size() / type_اندازه;
        GGML_ASSERT(is_array || ne == 1);
        أرجع ne;
    }

    قالب <نام_نوع T>
    ثابت T & get_val(ثابت size_t i = 0) ثابت {
        GGML_ASSERT(type_مقصد_gguf_type<T>::value == type);
        إذا constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(داده_string.size() >= i+1);
            أرجع داده_string[i];
        }
        ثابت size_t type_اندازه = gguf_type_اندازه(type);
        GGML_ASSERT(داده.size() % type_اندازه == 0);
        GGML_ASSERT(داده.size() >= (i+1)*type_اندازه);
        أرجع reinterpret_cast<ثابت T *>(داده.داده())[i];
    }

    فراغ cast(ثابت تعداد gguf_type new_type) {
        ثابت size_t new_type_اندازه = gguf_type_اندازه(new_type);
        GGML_ASSERT(داده.size() % new_type_اندازه == 0);
        type = new_type;
    }
};

بنية gguf_tensor_info {
    بنية ggml_tensor t; // for holding the equivalent info
    uint64_t جابه_جایی;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

بنية gguf_context {
    uint32_t نسخه_2 = GGUF_نسخه;

    std::vector<بنية gguf_kv> kv;
    std::vector<بنية gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t جابه_جایی    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    فراغ * داده = اشاره_تهی;
};

بنية gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            فراغ * userdata,
            size_t max_chunk_read,
            uint64_t داده_جابه_جایی = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          شروع_جابه_جایی(داده_جابه_جایی),
          داده_جابه_جایی(داده_جابه_جایی),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    ساكن uint64_t file_remain(FILE * file) {
        ثابت int64_t cur = gguf_ftell(file);
        إذا (cur < 0) {
            أرجع 0;
        }
        إذا (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            أرجع 0;
        }
        ثابت int64_t end = gguf_ftell(file);
        إذا (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            أرجع 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        أرجع static_cast<uint64_t>(end - cur);
    }

    قالب <نام_نوع T>
    منطقي اقرأ(T & dst) ثابت {
        ثابت size_t size = حجم(dst);
        إذا (size > nbytes_remain) {
            أرجع خطأ;
        }
        أرجع read_raw(&dst, size) == size;
    }

    قالب <نام_نوع T>
    منطقي اقرأ(std::vector<T> & dst, ثابت size_t n) ثابت {
        إذا (n > GGUF_MAX_ARRAY_ELEMENTS) {
            أرجع خطأ;
        }
        إذا constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            إذا (n > SIZE_MAX / حجم(uint64_t)) {
                أرجع خطأ;
            }
            إذا (nbytes_remain < n * حجم(uint64_t)) {
                أرجع خطأ;
            }
        } وإلا {
            إذا (n > SIZE_MAX / حجم(T)) {
                أرجع خطأ;
            }
            إذا (nbytes_remain < n * حجم(T)) {
                أرجع خطأ;
            }
        }
        dst.resize(n);
        لكل (size_t i = 0; i < dst.size(); ++i) {
            إذا constexpr (std::is_same<T, منطقي>::value) {
                منطقي tmp;
                إذا (!اقرأ(tmp)) {
                    أرجع خطأ;
                }
                dst[i] = tmp;
            } وإلا {
                إذا (!اقرأ(dst[i])) {
                    أرجع خطأ;
                }
            }
        }
        أرجع درست;
    }

    منطقي اقرأ(منطقي & dst) ثابت {
        int8_t tmp = -1;
        إذا (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = tmp != 0;
        أرجع درست;
    }

    منطقي اقرأ(تعداد ggml_type & dst) ثابت {
        int32_t tmp = -1;
        إذا (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = ggml_type(tmp);
        أرجع درست;
    }

    منطقي اقرأ(تعداد gguf_type & dst) ثابت {
        int32_t tmp = -1;
        إذا (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = gguf_type(tmp);
        أرجع درست;
    }

    منطقي اقرأ(std::string & dst) ثابت {
        uint64_t size = 0;
        إذا (!اقرأ(size)) {
            أرجع خطأ;
        }
        إذا (size > GGUF_MAX_STRING_درازی) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_درازی);
            أرجع خطأ;
        }
        إذا (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            أرجع خطأ;
        }
        dst.resize(static_cast<size_t>(size));
        أرجع read_raw(dst.داده(), static_cast<size_t>(size)) == size;
    }

    منطقي اقرأ(فراغ * dst, ثابت size_t size) ثابت {
        إذا (size > nbytes_remain) {
            أرجع خطأ;
        }
        أرجع read_raw(dst, size) == size;
    }

    uint64_t tell() ثابت {
        أرجع داده_جابه_جایی;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ثابت {
        أرجع شروع_جابه_جایی;
    }

    منطقي seek(uint64_t absolute_جابه_جایی) ثابت {
        ثابت uint64_t انتها_جابه_جایی = uint64_t(داده_جابه_جایی) + nbytes_remain;
        إذا (absolute_جابه_جایی > انتها_جابه_جایی) {
            أرجع خطأ;
        }

        داده_جابه_جایی = absolute_جابه_جایی;
        nbytes_remain = انتها_جابه_جایی - absolute_جابه_جایی;

        أرجع درست;
    }

خاص:
    size_t read_raw(فراغ * dst, size_t size) ثابت {
        إذا (callback == اشاره_تهی || size == 0) {
            أرجع 0;
        }

        uint8_t * داده = static_cast<uint8_t *>(dst);
        size_t مجموع_nread = 0;
        منطقي reached_eof = خطأ;

        طالما (مجموع_nread < size) {
            ثابت size_t chunk_اندازه = std::min(max_chunk_read, size - مجموع_nread);
            إذا (داده_جابه_جایی + مجموع_nread < داده_جابه_جایی) {
                اكسر;
            }
            ثابت size_t nread = callback(userdata, static_cast<فراغ *>(داده + مجموع_nread), داده_جابه_جایی + مجموع_nread, chunk_اندازه);
            مجموع_nread += nread;
            إذا (nread != chunk_اندازه) {
                reached_eof = درست;
                اكسر;
            }
        }

        داده_جابه_جایی += مجموع_nread;
        GGML_ASSERT(مجموع_nread <= nbytes_remain);
        nbytes_remain -= مجموع_nread;

        إذا (reached_eof) {
            nbytes_remain = 0;
        }

        أرجع مجموع_nread;
    }

    gguf_reader_callback_t callback = اشاره_تهی;
    فراغ * userdata = اشاره_تهی;
    size_t max_chunk_read = 0;
    uint64_t شروع_جابه_جایی = 0;
    تغییرپذیر uint64_t داده_جابه_جایی = 0;
    تغییرپذیر uint64_t nbytes_remain = 0;
};

بنية gguf_context * gguf_init_خالی(فراغ) {
    أرجع جديد gguf_context;
}

قالب<نام_نوع T>
منطقي gguf_read_emplace_helper(ثابت بنية gguf_reader & gr, std::vector<بنية gguf_kv> & kv, ثابت std::string & کلید, ثابت منطقي is_array, ثابت size_t n) {
    إذا (is_array) {
        std::vector<T> value;
        حاول {
            إذا (!gr.اقرأ(value, n)) {
                أرجع خطأ;
            }
        } امسك (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, کلید.c_str());
            أرجع خطأ;
        } امسك (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, کلید.c_str());
            أرجع خطأ;
        }
        kv.emplace_back(کلید, value);
    } وإلا {
        T value;
        إذا (!gr.اقرأ(value)) {
            أرجع خطأ;
        }
        kv.emplace_back(کلید, value);
    }
    أرجع درست;
}

ساكن بنية gguf_context * gguf_init_from_reader(ثابت بنية gguf_reader & gr, بنية gguf_init_params params) {
    بنية gguf_context * ctx = جديد gguf_context;

    منطقي ok = درست;

    // file magic
    {
        std::vector<حرف> magic;
        ok = ok && gr.اقرأ(magic, 4);

        إذا (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            أرجع اشاره_تهی;
        }

        لكل (uint32_t i = 0; i < magic.size(); i++) {
            إذا (magic[i] != GGUF_MAGIC[i]) {
                حرف c0 = isprint(magic[0]) ? magic[0] : '?';
                حرف c1 = isprint(magic[1]) ? magic[1] : '?';
                حرف c2 = isprint(magic[2]) ? magic[2] : '?';
                حرف c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }
        }
    }

    //سرآیند
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    إذا (ok && gr.اقرأ(ctx->نسخه_2)) {
        إذا (ok && ctx->نسخه_2 == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->نسخه_2);
            ok = خطأ;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        إذا (ok && (ctx->نسخه_2 & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->نسخه_2);
            ok = خطأ;
        }

        إذا (ok && ctx->نسخه_2 == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = خطأ;
        }
        إذا (ok && ctx->نسخه_2 > GGUF_نسخه) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->نسخه_2, GGUF_نسخه);
            ok = خطأ;
        }
    } وإلا {
        ok = خطأ;
    }

    إذا (ok && gr.اقرأ(n_tensors)) {
        static_assert(حجم(size_t) <= 8 && حجم(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        إذا (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/حجم(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/حجم(gguf_tensor_info));
            ok = خطأ;
        }
    } وإلا {
        ok = خطأ;
    }

    إذا (ok && gr.اقرأ(n_kv)) {
        static_assert(حجم(size_t) <= 8 && حجم(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        إذا (n_kv < 0 || n_kv > int64_t(SIZE_MAX/حجم(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/حجم(gguf_kv));
            ok = خطأ;
        }
    } وإلا {
        ok = خطأ;
    }

    إذا (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        أرجع اشاره_تهی;
    }

    // KV pairs
    {
        std::unordered_set<std::string> دیده_کلیدها;

        لكل (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string کلید;
            gguf_type   type     = gguf_type(-1);
            منطقي        is_array = خطأ;
            uint64_t    n        = 1;

            حاول {
                ok = ok && gr.اقرأ(کلید);
            } امسك (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            } امسك (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            }
            إذا (ok && کلید.خالی()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = خطأ;
            }
            إذا (ok && !دیده_کلیدها.insert(کلید).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, کلید.c_str(), i);
                ok = خطأ;
            }
            إذا (!ok) {
                اكسر;
            }

            ok = ok && gr.اقرأ(type);
            إذا (type == GGUF_TYPE_ARRAY) {
                is_array = درست;
                ok = ok && gr.اقرأ(type);
                ok = ok && gr.اقرأ(n);
            }
            إذا (!ok) {
                اكسر;
            }

            بدل (type) {
                حالة GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<عائم>      (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<منطقي>       (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<مزدوج>     (gr, ctx->kv, کلید, is_array, n); اكسر;
                حالة GGUF_TYPE_ARRAY:
                افتراضي:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, کلید.c_str(), type);
                        ok = خطأ;
                    } اكسر;
            }
        }

        إذا (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            أرجع اشاره_تهی;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ثابت صحيح alignment_idx = gguf_پیدا_کن_کلید(ctx, GGUF_کلید_GENERAL_ALIGNMENT);
        إذا (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_کلید_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            أرجع اشاره_تهی;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        إذا (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            أرجع اشاره_تهی;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> دیده_tensor_نامها;

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
            إذا (name.درازی() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.درازی(), GGML_MAX_NAME);
                ok = خطأ;
                اكسر;
            }
            ggml_مجموعه_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            إذا (ok && !دیده_tensor_نامها.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = خطأ;
                اكسر;
            }
        }
        إذا (!ok) {
            اكسر;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.اقرأ(n_dims);
            إذا (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = خطأ;
                اكسر;
            }
            لكل (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                إذا (j < n_dims) {
                    ok = ok && gr.اقرأ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                إذا (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = خطأ;
                    اكسر;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            إذا (ok) {
                int64_t current_nelements = info.t.ne[0];
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                    إذا (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        إذا (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = خطأ;
                             اكسر;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 إذا (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     اكسر;
                 }
            }
        }
        إذا (!ok) {
            اكسر;
        }

        // tensor type
        {
            ok = ok && gr.اقرأ(info.t.type);

            // check that tensor type is within defined range
            إذا (info.t.type < 0 || info.t.type >= GGML_TYPE_شمار) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_شمار);
                ok = خطأ;
                اكسر;
            }
            ثابت size_t  type_اندازه = ggml_type_اندازه(info.t.type);
            ثابت int64_t blck_اندازه = ggml_blck_اندازه(info.t.type);

            // check that row size is divisible by block size
            إذا (blck_اندازه == 0 || info.t.ne[0] % blck_اندازه != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (صحيح) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_اندازه);
                ok = خطأ;
                اكسر;
            }

            // check that the size of the tensor in bytes is representable
            إذا (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_اندازه(info.t.type)) > SIZE_MAX/ggml_type_اندازه(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = خطأ;
                اكسر;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_اندازه;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_اندازه);
            لكل (صحيح j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        إذا (!ok) {
            اكسر;
        }

        // tensor data offset within buffer
        ok = ok && gr.اقرأ(info.جابه_جایی);

        ctx->info.push_back(info);
    }

    إذا (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        أرجع اشاره_تهی;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    إذا (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        أرجع اشاره_تهی;
    }

    // store the current file offset - this is where the data section starts
    ctx->جابه_جایی = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        لكل (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت gguf_tensor_info & ti = ctx->info[i];
            إذا (ti.جابه_جایی != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.جابه_جایی, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }
            ثابت size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            إذا (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }
            ثابت size_t padded_اندازه = GGML_PAD(nbytes, ctx->alignment);
            إذا (SIZE_MAX - ctx->size < padded_اندازه) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_اندازه);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }
            ctx->size += padded_اندازه;
        }
    }

    // load the tensor data only if requested
    إذا (params.ctx != اشاره_تهی) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_اندازه = 0;
        إذا (params.no_alloc) {
            إذا (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }

            ثابت size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_اندازه = overhead;
        } وإلا {
            إذا ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }

            ثابت size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            إذا (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }

            mem_اندازه = overhead + ctx->size;
        }

        بنية ggml_init_params pdata = {
            /*mem_size   =*/ mem_اندازه,
            /*mem_buffer =*/ اشاره_تهی,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        إذا (*params.ctx == اشاره_تهی) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            أرجع اشاره_تهی;
        }

        بنية ggml_context * ctx_داده = *params.ctx;

        بنية ggml_tensor * داده = اشاره_تهی;

        إذا (!params.no_alloc) {
            داده = ggml_new_tensor_1d(ctx_داده, GGML_TYPE_I8, ctx->size);

            ok = ok && داده != اشاره_تهی;

            إذا (ok) {
                ggml_مجموعه_name(داده, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.اقرأ(داده->داده, ctx->size);

            إذا (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_داده);
                *params.ctx = اشاره_تهی;
                gguf_free(ctx);
                أرجع اشاره_تهی;
            }

            ctx->داده = داده->داده;
        }

        ggml_مجموعه_no_alloc(ctx_داده, درست);

        // create the tensors
        لكل (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت بنية gguf_tensor_info & info = ctx->info[i];

            بنية ggml_tensor * cur = ggml_new_tensor(ctx_داده, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != اشاره_تهی;

            إذا (!ok) {
                اكسر;
            }

            ggml_مجموعه_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            إذا (!params.no_alloc) {
                cur->داده = (حرف *) داده->داده + info.جابه_جایی;
            }
        }

        إذا (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_داده);
            *params.ctx = اشاره_تهی;
            gguf_free(ctx);
            أرجع اشاره_تهی;
        }

        ggml_مجموعه_no_alloc(ctx_داده, params.no_alloc);
    }

    أرجع ctx;
}

بنية gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, فراغ * userdata, size_t max_chunk_read, uint64_t max_expected_اندازه, بنية gguf_init_params params) {
    إذا (callback == اشاره_تهی) {
        أرجع اشاره_تهی;
    }

    ثابت بنية gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_اندازه);
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_file_reader {
    FILE * file;
    uint64_t جابه_جایی;
};

ساكن size_t gguf_file_reader_callback(فراغ * userdata, فراغ * output, uint64_t جابه_جایی, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    إذا (reader.جابه_جایی != جابه_جایی) {
        إذا (جابه_جایی > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(جابه_جایی), SEEK_SET) != 0) {
            أرجع 0;
        }

        reader.جابه_جایی = جابه_جایی;
    }

    ثابت size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.جابه_جایی += nread;
    أرجع nread;
}

بنية gguf_context * gguf_init_from_file_ptr(FILE * file, بنية gguf_init_params params) {
    إذا (!file) {
        أرجع اشاره_تهی;
    }

    ثابت int64_t cur = gguf_ftell(file);
    إذا (cur < 0) {
        أرجع اشاره_تهی;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ثابت بنية gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.جابه_جایی, gguf_reader::file_remain(file));
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_میانگیر_reader {
    ثابت uint8_t * داده;
    size_t          size;
};

ساكن size_t gguf_میانگیر_reader_callback(فراغ * userdata, فراغ * output, uint64_t جابه_جایی, size_t len) {
    GGML_ASSERT(len > 0);

    ثابت gguf_میانگیر_reader & reader = *static_cast<gguf_میانگیر_reader *>(userdata);

    إذا (جابه_جایی > reader.size || len > reader.size - جابه_جایی) {
        أرجع 0;
    }

    ثابت size_t داده_جابه_جایی = static_cast<size_t>(جابه_جایی);
    ثابت size_t nread = std::min(len, reader.size - داده_جابه_جایی);
    کپی_حافظه(static_cast<uint8_t *>(output), reader.داده + داده_جابه_جایی, nread);
    أرجع nread;
}

بنية gguf_context * gguf_init_from_میانگیر(ثابت فراغ * داده, size_t size, بنية gguf_init_params params) {
    إذا (داده == اشاره_تهی || size == 0) {
        أرجع اشاره_تهی;
    }

    gguf_میانگیر_reader reader = {
        /*.data = */ static_cast<ثابت uint8_t *>(داده),
        /*.size = */ size,
    };
    ثابت بنية gguf_reader gr(gguf_میانگیر_reader_callback, &reader, SIZE_MAX, 0, size);
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_context * gguf_init_from_file(ثابت حرف * fname, بنية gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    إذا (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        أرجع اشاره_تهی;
    }

    بنية gguf_context * نتیجه = gguf_init_from_file_ptr(file, params);
    fclose(file);
    أرجع نتیجه;
}

فراغ gguf_free(بنية gguf_context * ctx) {
    إذا (ctx == اشاره_تهی) {
        أرجع;
    }
    حذف ctx;
}

ثابت حرف * gguf_type_name(تعداد gguf_type type) {
    خودکار it = GGUF_TYPE_NAME.find(type);
    أرجع it == GGUF_TYPE_NAME.end() ? اشاره_تهی : it->second;
}

uint32_t gguf_get_نسخه(ثابت بنية gguf_context * ctx) {
    أرجع ctx->نسخه_2;
}

size_t gguf_get_alignment(ثابت بنية gguf_context * ctx) {
    أرجع ctx->alignment;
}

size_t gguf_get_داده_جابه_جایی(ثابت بنية gguf_context * ctx) {
    أرجع ctx->جابه_جایی;
}

int64_t gguf_get_n_kv(ثابت بنية gguf_context * ctx) {
    أرجع ctx->kv.size();
}

int64_t gguf_پیدا_کن_کلید(ثابت بنية gguf_context * ctx, ثابت حرف * کلید) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ثابت int64_t n_kv = gguf_get_n_kv(ctx);

    لكل (int64_t i = 0; i < n_kv; ++i) {
        إذا (strcmp(کلید, gguf_get_کلید(ctx, i)) == 0) {
            keyfound = i;
            اكسر;
        }
    }

    أرجع keyfound;
}

ثابت حرف * gguf_get_کلید(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    أرجع ctx->kv[کلید_id].get_کلید().c_str();
}

تعداد gguf_type gguf_get_kv_type(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    أرجع ctx->kv[کلید_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[کلید_id].get_type();
}

تعداد gguf_type gguf_get_arr_type(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].is_array);
    أرجع ctx->kv[کلید_id].get_type();
}

ثابت فراغ * gguf_get_arr_داده(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_type() != GGUF_TYPE_STRING);
    أرجع ctx->kv[کلید_id].داده.داده();
}

ثابت حرف * gguf_get_arr_str(ثابت بنية gguf_context * ctx, int64_t کلید_id, size_t i) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_type() == GGUF_TYPE_STRING);
    أرجع ctx->kv[کلید_id].داده_string[i].c_str();
}

size_t gguf_get_arr_n(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));

    إذا (ctx->kv[کلید_id].type == GGUF_TYPE_STRING) {
        أرجع ctx->kv[کلید_id].داده_string.size();
    }

    ثابت size_t type_اندازه = gguf_type_اندازه(ctx->kv[کلید_id].type);
    GGML_ASSERT(ctx->kv[کلید_id].داده.size() % type_اندازه == 0);
    أرجع ctx->kv[کلید_id].داده.size() / type_اندازه;
}

uint8_t gguf_get_val_u8(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<int32_t>();
}

عائم gguf_get_val_f32(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<عائم>();
}

uint64_t gguf_get_val_u64(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<int64_t>();
}

مزدوج gguf_get_val_f64(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<مزدوج>();
}

منطقي gguf_get_val_bool(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<منطقي>();
}

ثابت حرف * gguf_get_val_str(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    أرجع ctx->kv[کلید_id].get_val<std::string>().c_str();
}

ثابت فراغ * gguf_get_val_داده(ثابت بنية gguf_context * ctx, int64_t کلید_id) {
    GGML_ASSERT(کلید_id >= 0 && کلید_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلید_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[کلید_id].get_type() != GGUF_TYPE_STRING);
    أرجع ctx->kv[کلید_id].داده.داده();
}

int64_t gguf_get_n_tensors(ثابت بنية gguf_context * ctx) {
    أرجع ctx->info.size();
}

int64_t gguf_پیدا_کن_tensor(ثابت بنية gguf_context * ctx, ثابت حرف * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    لكل (int64_t i = 0; i < n_tensors; ++i) {
        إذا (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            اكسر;
        }
    }

    أرجع tensor_id;
}

size_t gguf_get_tensor_جابه_جایی(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ctx->info[tensor_id].جابه_جایی;
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

size_t gguf_get_tensor_اندازه(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_کلید(بنية gguf_context * ctx, ثابت حرف * کلید) {
    ثابت int64_t کلید_id = gguf_پیدا_کن_کلید(ctx, کلید);
    إذا (کلید_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + کلید_id);
    }
    أرجع کلید_id;
}

قالب<نام_نوع T>
ساكن فراغ gguf_check_reserved_کلیدها(ثابت std::string & کلید, ثابت T val) {
    إذا (کلید == GGUF_کلید_GENERAL_ALIGNMENT) {
        إذا constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_کلید_GENERAL_ALIGNMENT " must be power of 2");
        } وإلا {
            GGML_استفادهنشده(val);
            GGML_ABORT(GGUF_کلید_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

فراغ gguf_مجموعه_val_u8(بنية gguf_context * ctx, ثابت حرف * کلید, uint8_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_i8(بنية gguf_context * ctx, ثابت حرف * کلید, int8_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_u16(بنية gguf_context * ctx, ثابت حرف * کلید, uint16_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_i16(بنية gguf_context * ctx, ثابت حرف * کلید, int16_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_u32(بنية gguf_context * ctx, ثابت حرف * کلید, uint32_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_i32(بنية gguf_context * ctx, ثابت حرف * کلید, int32_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_f32(بنية gguf_context * ctx, ثابت حرف * کلید, عائم val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_u64(بنية gguf_context * ctx, ثابت حرف * کلید, uint64_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_i64(بنية gguf_context * ctx, ثابت حرف * کلید, int64_t val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_f64(بنية gguf_context * ctx, ثابت حرف * کلید, مزدوج val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_bool(بنية gguf_context * ctx, ثابت حرف * کلید, منطقي val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, val);
}

فراغ gguf_مجموعه_val_str(بنية gguf_context * ctx, ثابت حرف * کلید, ثابت حرف * val) {
    gguf_check_reserved_کلیدها(کلید, val);
    gguf_remove_کلید(ctx, کلید);
    ctx->kv.emplace_back(کلید, std::string(val));
}

فراغ gguf_مجموعه_arr_داده(بنية gguf_context * ctx, ثابت حرف * کلید, تعداد gguf_type type, ثابت فراغ * داده, size_t n) {
    gguf_check_reserved_کلیدها(کلید, داده);
    gguf_remove_کلید(ctx, کلید);

    ثابت size_t nbytes = n*gguf_type_اندازه(type);
    std::vector<int8_t> tmp(nbytes);
    إذا (!tmp.خالی()) {
        کپی_حافظه(tmp.داده(), داده, nbytes);
    }
    ctx->kv.emplace_back(کلید, tmp);
    ctx->kv.back().cast(type);
}

فراغ gguf_مجموعه_arr_str(بنية gguf_context * ctx, ثابت حرف * کلید, ثابت حرف ** داده, size_t n) {
    gguf_check_reserved_کلیدها(کلید, داده);
    gguf_remove_کلید(ctx, کلید);

    std::vector<std::string> tmp(n);
    لكل (size_t i = 0; i < n; ++i) {
        tmp[i] = داده[i];
    }
    ctx->kv.emplace_back(کلید, tmp);
}

// set or add KV pairs from another context
فراغ gguf_مجموعه_kv(بنية gguf_context * ctx, ثابت بنية gguf_context * src) {
    ثابت int64_t n_kv = gguf_get_n_kv(src);
    لكل (int64_t i = 0; i < n_kv; ++i) {
        ثابت بنية gguf_kv & kv = src->kv[i];

        إذا (!kv.is_array) {
            بدل (kv.get_type()) {
                حالة GGUF_TYPE_UINT8:   gguf_مجموعه_val_u8  (ctx, kv.get_کلید().c_str(), kv.get_val<uint8_t>());             اكسر;
                حالة GGUF_TYPE_INT8:    gguf_مجموعه_val_i8  (ctx, kv.get_کلید().c_str(), kv.get_val<int8_t>());              اكسر;
                حالة GGUF_TYPE_UINT16:  gguf_مجموعه_val_u16 (ctx, kv.get_کلید().c_str(), kv.get_val<uint16_t>());            اكسر;
                حالة GGUF_TYPE_INT16:   gguf_مجموعه_val_i16 (ctx, kv.get_کلید().c_str(), kv.get_val<int16_t>());             اكسر;
                حالة GGUF_TYPE_UINT32:  gguf_مجموعه_val_u32 (ctx, kv.get_کلید().c_str(), kv.get_val<uint32_t>());            اكسر;
                حالة GGUF_TYPE_INT32:   gguf_مجموعه_val_i32 (ctx, kv.get_کلید().c_str(), kv.get_val<int32_t>());             اكسر;
                حالة GGUF_TYPE_FLOAT32: gguf_مجموعه_val_f32 (ctx, kv.get_کلید().c_str(), kv.get_val<عائم>());               اكسر;
                حالة GGUF_TYPE_UINT64:  gguf_مجموعه_val_u64 (ctx, kv.get_کلید().c_str(), kv.get_val<uint64_t>());            اكسر;
                حالة GGUF_TYPE_INT64:   gguf_مجموعه_val_i64 (ctx, kv.get_کلید().c_str(), kv.get_val<int64_t>());             اكسر;
                حالة GGUF_TYPE_FLOAT64: gguf_مجموعه_val_f64 (ctx, kv.get_کلید().c_str(), kv.get_val<مزدوج>());              اكسر;
                حالة GGUF_TYPE_BOOL:    gguf_مجموعه_val_bool(ctx, kv.get_کلید().c_str(), kv.get_val<منطقي>());                اكسر;
                حالة GGUF_TYPE_STRING:  gguf_مجموعه_val_str (ctx, kv.get_کلید().c_str(), kv.get_val<std::string>().c_str()); اكسر;
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
                gguf_مجموعه_arr_داده(ctx, kv.get_کلید().c_str(), kv.get_type(), kv.داده.داده(), ne);
            } اكسر;
            حالة GGUF_TYPE_STRING: {
                std::vector<ثابت حرف *> tmp(ne);
                لكل (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.داده_string[j].c_str();
                }
                gguf_مجموعه_arr_str(ctx, kv.get_کلید().c_str(), tmp.داده(), ne);
            } اكسر;
            حالة GGUF_TYPE_ARRAY:
            افتراضي: GGML_ABORT("invalid type");
        }
    }
}

فراغ gguf_جمع_کن_tensor(
             بنية gguf_context * ctx,
        ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    إذا (gguf_پیدا_کن_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    بنية gguf_tensor_info ti;
    ti.t = *tensor;
    ti.جابه_جایی = ctx->info.خالی() ? 0 :
        ctx->info.back().جابه_جایی + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

فراغ gguf_مجموعه_tensor_type(بنية gguf_context * ctx, ثابت حرف * name, تعداد ggml_type type) {
    ثابت int64_t tensor_id = gguf_پیدا_کن_tensor(ctx, name);
    إذا (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    بنية ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ثابت size_t  type_اندازه = ggml_type_اندازه(type);
    ثابت int64_t blck_اندازه = ggml_blck_اندازه(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_اندازه == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_اندازه;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_اندازه);
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);
    لكل (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].جابه_جایی = ctx->info[i - 1].جابه_جایی + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

فراغ gguf_مجموعه_tensor_داده(بنية gguf_context * ctx, ثابت حرف * name, ثابت فراغ * داده) {
    ثابت int64_t tensor_id = gguf_پیدا_کن_tensor(ctx, name);
    إذا (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.داده = (فراغ *)(uintptr_t)داده; // double cast suppresses warning about casting away const
}

بنية gguf_writer_پایه {
    size_t written_بایتها {0u};

    مجازی ~gguf_writer_پایه(فراغ) = افتراضي;

    // we bet on devirtualization
    مجازی فراغ اكتب(int8_t val) = 0;
    مجازی فراغ اكتب(ثابت std::vector<int8_t> & val) = 0;
    مجازی فراغ write_tensor_داده(ثابت بنية gguf_tensor_info & info, size_t جابه_جایی_داده, size_t alignment) = 0;

    قالب <نام_نوع T>
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
            ثابت uint64_t n = val.درازی();
            اكتب(n);
        }
        لكل (size_t i = 0; i < val.درازی(); ++i) {
            اكتب((val.داده())[i]);
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

        اكتب(kv.get_کلید());

        إذا (kv.is_array) {
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
                اكتب(kv.داده);
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
        اكتب(info.جابه_جایی);
    }

    فراغ pad(ثابت size_t alignment) {
        طالما (written_بایتها % alignment != 0) {
            ثابت int8_t zero = 0;
            اكتب(zero);
        }
    }
};

// vector buffer based writer
بنية gguf_writer_buf final : عام gguf_writer_پایه {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    استفاده gguf_writer_پایه::اكتب;

    فراغ اكتب(ثابت int8_t val) override {
        buf.push_back(val);
        written_بایتها++;
    }

    فراغ اكتب(ثابت std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_بایتها += val.size();
    }

    فراغ write_tensor_داده(ثابت بنية gguf_tensor_info & info, ثابت size_t جابه_جایی_داده, ثابت size_t alignment) override {
        GGML_ASSERT(buf.size() - جابه_جایی_داده == info.جابه_جایی);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t جابه_جایی = buf.size();
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(جابه_جایی + nbytes);
        إذا (info.t.میانگیر) {
            ggml_backend_tensor_get(&info.t, buf.داده() + جابه_جایی, 0, nbytes);
        } وإلا {
            GGML_ASSERT(info.t.داده);
            کپی_حافظه(buf.داده() + جابه_جایی, info.t.داده, nbytes);
        }
        written_بایتها += nbytes;

        pad(alignment);
    }
};

// file based writer
بنية gguf_writer_file final : عام gguf_writer_پایه {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    استفاده gguf_writer_پایه::اكتب;

    فراغ اكتب(ثابت int8_t val) override {
        ثابت خودکار real_val = static_cast<uint8_t>(val);
        ثابت خودکار ret = fputc(real_val, file);
        written_بایتها++;
        إذا (ret != real_val) {
            ارم std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((صحيح)real_val) + "'");
        }
    }

    فراغ اكتب(ثابت std::vector<int8_t> & val) override {
        ثابت خودکار ret = fwrite(val.داده(), 1, val.size(), file);
        written_بایتها += val.size();
        إذا (ret != val.size()) {
            ارم std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    فراغ write_tensor_داده(ثابت بنية gguf_tensor_info & info, ثابت size_t جابه_جایی_داده, ثابت size_t alignment) override {
        GGML_ASSERT(written_بایتها - جابه_جایی_داده == info.جابه_جایی);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        إذا (info.t.میانگیر) {
            ggml_backend_tensor_get(&info.t, buf.داده(), 0, nbytes);
        } وإلا {
            GGML_ASSERT(info.t.داده);
            کپی_حافظه(buf.داده(), info.t.داده, nbytes);
        }
        اكتب(buf);

        pad(alignment);
    }
};

قالب <نام_نوع writer_t>
ساكن فراغ gguf_write_بیرون(ثابت بنية gguf_context * ctx, writer_t & gw, منطقي only_meta) {
    ثابت int64_t n_kv      = gguf_get_n_kv(ctx);
    ثابت int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.اكتب(GGUF_MAGIC[0]);
    gw.اكتب(GGUF_MAGIC[1]);
    gw.اكتب(GGUF_MAGIC[2]);
    gw.اكتب(GGUF_MAGIC[3]);
    gw.اكتب(ctx->نسخه_2);
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

    إذا (only_meta) {
        أرجع;
    }

    ثابت size_t جابه_جایی_داده = gw.written_بایتها;

    // write tensor data
    لكل (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_داده(ctx->info[i], جابه_جایی_داده, ctx->alignment);
    }
}

فراغ gguf_write_مقصد_buf(ثابت بنية gguf_context * ctx, std::vector<int8_t> & buf, منطقي only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_بیرون(ctx, gw, only_meta);
}

منطقي gguf_write_مقصد_file_ptr(ثابت بنية gguf_context * ctx, FILE * file, منطقي only_meta) {
    GGML_ASSERT(file);

    حاول {
        gguf_writer_file gw(file);
        gguf_write_بیرون(ctx, gw, only_meta);
    } امسك (ثابت std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        أرجع خطأ;
    }
    أرجع درست;
}

منطقي gguf_write_مقصد_file(ثابت بنية gguf_context * ctx, ثابت حرف * fname, منطقي only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    إذا (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        أرجع خطأ;
    }

    ثابت منطقي success = gguf_write_مقصد_file_ptr(ctx, file, only_meta);
    إذا (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    أرجع success;
}

size_t gguf_get_meta_اندازه(ثابت بنية gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_مقصد_buf(ctx, buf, /*only_meta =*/ درست);
    أرجع buf.size();
}

فراغ gguf_get_meta_داده(ثابت بنية gguf_context * ctx, فراغ * داده) {
    std::vector<int8_t> buf;
    gguf_write_مقصد_buf(ctx, buf, /*only_meta =*/ درست);
    کپی_حافظه(داده, buf.داده(), buf.size());
}
