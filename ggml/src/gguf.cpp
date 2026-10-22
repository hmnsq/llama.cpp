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

#تعریف_کړه GGUF_MAX_STRING_اوږدتیا  (1024*1024*1024)
#تعریف_کړه GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#که_تعریف_شوی _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#وإلا
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#د_که_پای

کينډۍ <ډول_نوم T>
بنية type_منزل_gguf_type;

کينډۍ <>
بنية type_منزل_gguf_type<uint8_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT8;
};

کينډۍ <>
بنية type_منزل_gguf_type<int8_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT8;
};

کينډۍ <>
بنية type_منزل_gguf_type<uint16_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT16;
};

کينډۍ <>
بنية type_منزل_gguf_type<int16_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT16;
};

کينډۍ <>
بنية type_منزل_gguf_type<uint32_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT32;
};

کينډۍ <>
بنية type_منزل_gguf_type<int32_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT32;
};

کينډۍ <>
بنية type_منزل_gguf_type<عائم> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_FLOAT32;
};

کينډۍ <>
بنية type_منزل_gguf_type<منطقي> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_BOOL;
};

کينډۍ <>
بنية type_منزل_gguf_type<std::string> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_STRING;
};

کينډۍ <>
بنية type_منزل_gguf_type<uint64_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_UINT64;
};

کينډۍ <>
بنية type_منزل_gguf_type<int64_t> {
    ساكن constexpr تعداد gguf_type value = GGUF_TYPE_INT64;
};

کينډۍ <>
بنية type_منزل_gguf_type<مزدوج> {
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
static_assert(GGUF_TYPE_شمېر == 13, "GGUF_TYPE_COUNT != 13");

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
static_assert(GGUF_TYPE_شمېر == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_اندازه(تعداد gguf_type type) {
    خپلکاری it = GGUF_TYPE_اندازه.find(type);
    أرجع it == GGUF_TYPE_اندازه.end() ? 0 : it->second;
}

بنية gguf_kv {
    std::string کلی;

    منطقي is_array;
    تعداد gguf_type type;

    std::vector<int8_t>      معلومات;
    std::vector<std::string> معلومات_string;

    کينډۍ <ډول_نوم T>
    gguf_kv(ثابت std::string & کلی, ثابت T value)
            : کلی(کلی), is_array(خطأ), type(type_منزل_gguf_type<T>::value) {
        GGML_ASSERT(!کلی.تشه());
        معلومات.resize(حجم(T));
        حافظه_نقل(معلومات.معلومات(), &value, حجم(T));
    }

    کينډۍ <ډول_نوم T>
    gguf_kv(ثابت std::string & کلی, ثابت std::vector<T> & value)
            : کلی(کلی), is_array(ریښتیا), type(type_منزل_gguf_type<T>::value) {
        GGML_ASSERT(!کلی.تشه());
        معلومات.resize(value.size()*حجم(T));
        لكل (size_t i = 0; i < value.size(); ++i) {
            ثابت T tmp = value[i];
            حافظه_نقل(معلومات.معلومات() + i*حجم(T), &tmp, حجم(T));
        }
    }

    gguf_kv(ثابت std::string & کلی, ثابت std::string & value)
            : کلی(کلی), is_array(خطأ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلی.تشه());
        معلومات_string.push_back(value);
    }

    gguf_kv(ثابت std::string & کلی, ثابت std::vector<std::string> & value)
            : کلی(کلی), is_array(ریښتیا), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!کلی.تشه());
        معلومات_string = value;
    }

    ثابت std::string & get_کلی() ثابت {
        أرجع کلی;
    }

    ثابت تعداد gguf_type & get_type() ثابت {
        أرجع type;
    }

    size_t get_ne() ثابت {
        إذا (type == GGUF_TYPE_STRING) {
            ثابت size_t ne = معلومات_string.size();
            GGML_ASSERT(is_array || ne == 1);
            أرجع ne;
        }
        ثابت size_t type_اندازه = gguf_type_اندازه(type);
        GGML_ASSERT(معلومات.size() % type_اندازه == 0);
        ثابت size_t ne = معلومات.size() / type_اندازه;
        GGML_ASSERT(is_array || ne == 1);
        أرجع ne;
    }

    کينډۍ <ډول_نوم T>
    ثابت T & get_val(ثابت size_t i = 0) ثابت {
        GGML_ASSERT(type_منزل_gguf_type<T>::value == type);
        إذا constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(معلومات_string.size() >= i+1);
            أرجع معلومات_string[i];
        }
        ثابت size_t type_اندازه = gguf_type_اندازه(type);
        GGML_ASSERT(معلومات.size() % type_اندازه == 0);
        GGML_ASSERT(معلومات.size() >= (i+1)*type_اندازه);
        أرجع reinterpret_cast<ثابت T *>(معلومات.معلومات())[i];
    }

    فراغ cast(ثابت تعداد gguf_type new_type) {
        ثابت size_t new_type_اندازه = gguf_type_اندازه(new_type);
        GGML_ASSERT(معلومات.size() % new_type_اندازه == 0);
        type = new_type;
    }
};

بنية gguf_tensor_info {
    بنية ggml_tensor t; // for holding the equivalent info
    uint64_t دننه_واټن;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

بنية gguf_context {
    uint32_t نسخه_2 = GGUF_نسخه;

    std::vector<بنية gguf_kv> kv;
    std::vector<بنية gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t دننه_واټن    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    فراغ * معلومات = تش_اشاره;
};

بنية gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            فراغ * userdata,
            size_t max_chunk_read,
            uint64_t معلومات_دننه_واټن = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          پیل_دننه_واټن(معلومات_دننه_واټن),
          معلومات_دننه_واټن(معلومات_دننه_واټن),
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

    کينډۍ <ډول_نوم T>
    منطقي اقرأ(T & dst) ثابت {
        ثابت size_t size = حجم(dst);
        إذا (size > nbytes_remain) {
            أرجع خطأ;
        }
        أرجع read_raw(&dst, size) == size;
    }

    کينډۍ <ډول_نوم T>
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
        أرجع ریښتیا;
    }

    منطقي اقرأ(منطقي & dst) ثابت {
        int8_t tmp = -1;
        إذا (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = tmp != 0;
        أرجع ریښتیا;
    }

    منطقي اقرأ(تعداد ggml_type & dst) ثابت {
        int32_t tmp = -1;
        إذا (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = ggml_type(tmp);
        أرجع ریښتیا;
    }

    منطقي اقرأ(تعداد gguf_type & dst) ثابت {
        int32_t tmp = -1;
        إذا (!اقرأ(tmp)) {
            أرجع خطأ;
        }
        dst = gguf_type(tmp);
        أرجع ریښتیا;
    }

    منطقي اقرأ(std::string & dst) ثابت {
        uint64_t size = 0;
        إذا (!اقرأ(size)) {
            أرجع خطأ;
        }
        إذا (size > GGUF_MAX_STRING_اوږدتیا) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_اوږدتیا);
            أرجع خطأ;
        }
        إذا (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            أرجع خطأ;
        }
        dst.resize(static_cast<size_t>(size));
        أرجع read_raw(dst.معلومات(), static_cast<size_t>(size)) == size;
    }

    منطقي اقرأ(فراغ * dst, ثابت size_t size) ثابت {
        إذا (size > nbytes_remain) {
            أرجع خطأ;
        }
        أرجع read_raw(dst, size) == size;
    }

    uint64_t tell() ثابت {
        أرجع معلومات_دننه_واټن;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ثابت {
        أرجع پیل_دننه_واټن;
    }

    منطقي seek(uint64_t absolute_دننه_واټن) ثابت {
        ثابت uint64_t پای_ټکی_دننه_واټن = uint64_t(معلومات_دننه_واټن) + nbytes_remain;
        إذا (absolute_دننه_واټن > پای_ټکی_دننه_واټن) {
            أرجع خطأ;
        }

        معلومات_دننه_واټن = absolute_دننه_واټن;
        nbytes_remain = پای_ټکی_دننه_واټن - absolute_دننه_واټن;

        أرجع ریښتیا;
    }

خاص:
    size_t read_raw(فراغ * dst, size_t size) ثابت {
        إذا (callback == تش_اشاره || size == 0) {
            أرجع 0;
        }

        uint8_t * معلومات = static_cast<uint8_t *>(dst);
        size_t ټول_nread = 0;
        منطقي reached_eof = خطأ;

        طالما (ټول_nread < size) {
            ثابت size_t chunk_اندازه = std::min(max_chunk_read, size - ټول_nread);
            إذا (معلومات_دننه_واټن + ټول_nread < معلومات_دننه_واټن) {
                اكسر;
            }
            ثابت size_t nread = callback(userdata, static_cast<فراغ *>(معلومات + ټول_nread), معلومات_دننه_واټن + ټول_nread, chunk_اندازه);
            ټول_nread += nread;
            إذا (nread != chunk_اندازه) {
                reached_eof = ریښتیا;
                اكسر;
            }
        }

        معلومات_دننه_واټن += ټول_nread;
        GGML_ASSERT(ټول_nread <= nbytes_remain);
        nbytes_remain -= ټول_nread;

        إذا (reached_eof) {
            nbytes_remain = 0;
        }

        أرجع ټول_nread;
    }

    gguf_reader_callback_t callback = تش_اشاره;
    فراغ * userdata = تش_اشاره;
    size_t max_chunk_read = 0;
    uint64_t پیل_دننه_واټن = 0;
    بدلیدونکی uint64_t معلومات_دننه_واټن = 0;
    بدلیدونکی uint64_t nbytes_remain = 0;
};

بنية gguf_context * gguf_init_تشه(فراغ) {
    أرجع جديد gguf_context;
}

کينډۍ<ډول_نوم T>
منطقي gguf_read_emplace_helper(ثابت بنية gguf_reader & gr, std::vector<بنية gguf_kv> & kv, ثابت std::string & کلی, ثابت منطقي is_array, ثابت size_t n) {
    إذا (is_array) {
        std::vector<T> value;
        حاول {
            إذا (!gr.اقرأ(value, n)) {
                أرجع خطأ;
            }
        } امسك (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, کلی.c_str());
            أرجع خطأ;
        } امسك (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, کلی.c_str());
            أرجع خطأ;
        }
        kv.emplace_back(کلی, value);
    } وإلا {
        T value;
        إذا (!gr.اقرأ(value)) {
            أرجع خطأ;
        }
        kv.emplace_back(کلی, value);
    }
    أرجع ریښتیا;
}

ساكن بنية gguf_context * gguf_init_from_reader(ثابت بنية gguf_reader & gr, بنية gguf_init_params params) {
    بنية gguf_context * ctx = جديد gguf_context;

    منطقي ok = ریښتیا;

    // file magic
    {
        std::vector<حرف> magic;
        ok = ok && gr.اقرأ(magic, 4);

        إذا (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            أرجع تش_اشاره;
        }

        لكل (uint32_t i = 0; i < magic.size(); i++) {
            إذا (magic[i] != GGUF_MAGIC[i]) {
                حرف c0 = isprint(magic[0]) ? magic[0] : '?';
                حرف c1 = isprint(magic[1]) ? magic[1] : '?';
                حرف c2 = isprint(magic[2]) ? magic[2] : '?';
                حرف c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }
        }
    }

    //سرلیک
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
        أرجع تش_اشاره;
    }

    // KV pairs
    {
        std::unordered_set<std::string> کتل_شوی_کلیګانې;

        لكل (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string کلی;
            gguf_type   type     = gguf_type(-1);
            منطقي        is_array = خطأ;
            uint64_t    n        = 1;

            حاول {
                ok = ok && gr.اقرأ(کلی);
            } امسك (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            } امسك (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = خطأ;
            }
            إذا (ok && کلی.تشه()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = خطأ;
            }
            إذا (ok && !کتل_شوی_کلیګانې.insert(کلی).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, کلی.c_str(), i);
                ok = خطأ;
            }
            إذا (!ok) {
                اكسر;
            }

            ok = ok && gr.اقرأ(type);
            إذا (type == GGUF_TYPE_ARRAY) {
                is_array = ریښتیا;
                ok = ok && gr.اقرأ(type);
                ok = ok && gr.اقرأ(n);
            }
            إذا (!ok) {
                اكسر;
            }

            بدل (type) {
                حالة GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<عائم>      (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<منطقي>       (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<مزدوج>     (gr, ctx->kv, کلی, is_array, n); اكسر;
                حالة GGUF_TYPE_ARRAY:
                افتراضي:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, کلی.c_str(), type);
                        ok = خطأ;
                    } اكسر;
            }
        }

        إذا (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            أرجع تش_اشاره;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ثابت صحيح alignment_idx = gguf_پیدا_کړه_کلی(ctx, GGUF_کلی_GENERAL_ALIGNMENT);
        إذا (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_کلی_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            أرجع تش_اشاره;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        إذا (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            أرجع تش_اشاره;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> کتل_شوی_tensor_نومونه;

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
            إذا (name.اوږدتیا() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.اوږدتیا(), GGML_MAX_NAME);
                ok = خطأ;
                اكسر;
            }
            ggml_سېټ_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            إذا (ok && !کتل_شوی_tensor_نومونه.insert(name).second) {
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
            إذا (info.t.type < 0 || info.t.type >= GGML_TYPE_شمېر) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_شمېر);
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
        ok = ok && gr.اقرأ(info.دننه_واټن);

        ctx->info.push_back(info);
    }

    إذا (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        أرجع تش_اشاره;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    إذا (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        أرجع تش_اشاره;
    }

    // store the current file offset - this is where the data section starts
    ctx->دننه_واټن = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        لكل (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت gguf_tensor_info & ti = ctx->info[i];
            إذا (ti.دننه_واټن != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.دننه_واټن, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }
            ثابت size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            إذا (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }
            ثابت size_t padded_اندازه = GGML_PAD(nbytes, ctx->alignment);
            إذا (SIZE_MAX - ctx->size < padded_اندازه) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_اندازه);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }
            ctx->size += padded_اندازه;
        }
    }

    // load the tensor data only if requested
    إذا (params.ctx != تش_اشاره) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_اندازه = 0;
        إذا (params.no_alloc) {
            إذا (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }

            ثابت size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_اندازه = overhead;
        } وإلا {
            إذا ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }

            ثابت size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            إذا (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                أرجع تش_اشاره;
            }

            mem_اندازه = overhead + ctx->size;
        }

        بنية ggml_init_params pdata = {
            /*mem_size   =*/ mem_اندازه,
            /*mem_buffer =*/ تش_اشاره,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        إذا (*params.ctx == تش_اشاره) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            أرجع تش_اشاره;
        }

        بنية ggml_context * ctx_معلومات = *params.ctx;

        بنية ggml_tensor * معلومات = تش_اشاره;

        إذا (!params.no_alloc) {
            معلومات = ggml_new_tensor_1d(ctx_معلومات, GGML_TYPE_I8, ctx->size);

            ok = ok && معلومات != تش_اشاره;

            إذا (ok) {
                ggml_سېټ_name(معلومات, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.اقرأ(معلومات->معلومات, ctx->size);

            إذا (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_معلومات);
                *params.ctx = تش_اشاره;
                gguf_free(ctx);
                أرجع تش_اشاره;
            }

            ctx->معلومات = معلومات->معلومات;
        }

        ggml_سېټ_no_alloc(ctx_معلومات, ریښتیا);

        // create the tensors
        لكل (size_t i = 0; i < ctx->info.size(); ++i) {
            ثابت بنية gguf_tensor_info & info = ctx->info[i];

            بنية ggml_tensor * cur = ggml_new_tensor(ctx_معلومات, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != تش_اشاره;

            إذا (!ok) {
                اكسر;
            }

            ggml_سېټ_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            إذا (!params.no_alloc) {
                cur->معلومات = (حرف *) معلومات->معلومات + info.دننه_واټن;
            }
        }

        إذا (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_معلومات);
            *params.ctx = تش_اشاره;
            gguf_free(ctx);
            أرجع تش_اشاره;
        }

        ggml_سېټ_no_alloc(ctx_معلومات, params.no_alloc);
    }

    أرجع ctx;
}

بنية gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, فراغ * userdata, size_t max_chunk_read, uint64_t max_expected_اندازه, بنية gguf_init_params params) {
    إذا (callback == تش_اشاره) {
        أرجع تش_اشاره;
    }

    ثابت بنية gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_اندازه);
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_file_reader {
    FILE * file;
    uint64_t دننه_واټن;
};

ساكن size_t gguf_file_reader_callback(فراغ * userdata, فراغ * output, uint64_t دننه_واټن, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    إذا (reader.دننه_واټن != دننه_واټن) {
        إذا (دننه_واټن > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(دننه_واټن), SEEK_SET) != 0) {
            أرجع 0;
        }

        reader.دننه_واټن = دننه_واټن;
    }

    ثابت size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.دننه_واټن += nread;
    أرجع nread;
}

بنية gguf_context * gguf_init_from_file_ptr(FILE * file, بنية gguf_init_params params) {
    إذا (!file) {
        أرجع تش_اشاره;
    }

    ثابت int64_t cur = gguf_ftell(file);
    إذا (cur < 0) {
        أرجع تش_اشاره;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ثابت بنية gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.دننه_واټن, gguf_reader::file_remain(file));
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_بفر_reader {
    ثابت uint8_t * معلومات;
    size_t          size;
};

ساكن size_t gguf_بفر_reader_callback(فراغ * userdata, فراغ * output, uint64_t دننه_واټن, size_t len) {
    GGML_ASSERT(len > 0);

    ثابت gguf_بفر_reader & reader = *static_cast<gguf_بفر_reader *>(userdata);

    إذا (دننه_واټن > reader.size || len > reader.size - دننه_واټن) {
        أرجع 0;
    }

    ثابت size_t معلومات_دننه_واټن = static_cast<size_t>(دننه_واټن);
    ثابت size_t nread = std::min(len, reader.size - معلومات_دننه_واټن);
    حافظه_نقل(static_cast<uint8_t *>(output), reader.معلومات + معلومات_دننه_واټن, nread);
    أرجع nread;
}

بنية gguf_context * gguf_init_from_بفر(ثابت فراغ * معلومات, size_t size, بنية gguf_init_params params) {
    إذا (معلومات == تش_اشاره || size == 0) {
        أرجع تش_اشاره;
    }

    gguf_بفر_reader reader = {
        /*.data = */ static_cast<ثابت uint8_t *>(معلومات),
        /*.size = */ size,
    };
    ثابت بنية gguf_reader gr(gguf_بفر_reader_callback, &reader, SIZE_MAX, 0, size);
    أرجع gguf_init_from_reader(gr, params);
}

بنية gguf_context * gguf_init_from_file(ثابت حرف * fname, بنية gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    إذا (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        أرجع تش_اشاره;
    }

    بنية gguf_context * پایله = gguf_init_from_file_ptr(file, params);
    fclose(file);
    أرجع پایله;
}

فراغ gguf_free(بنية gguf_context * ctx) {
    إذا (ctx == تش_اشاره) {
        أرجع;
    }
    ړنګول ctx;
}

ثابت حرف * gguf_type_name(تعداد gguf_type type) {
    خپلکاری it = GGUF_TYPE_NAME.find(type);
    أرجع it == GGUF_TYPE_NAME.end() ? تش_اشاره : it->second;
}

uint32_t gguf_get_نسخه(ثابت بنية gguf_context * ctx) {
    أرجع ctx->نسخه_2;
}

size_t gguf_get_alignment(ثابت بنية gguf_context * ctx) {
    أرجع ctx->alignment;
}

size_t gguf_get_معلومات_دننه_واټن(ثابت بنية gguf_context * ctx) {
    أرجع ctx->دننه_واټن;
}

int64_t gguf_get_n_kv(ثابت بنية gguf_context * ctx) {
    أرجع ctx->kv.size();
}

int64_t gguf_پیدا_کړه_کلی(ثابت بنية gguf_context * ctx, ثابت حرف * کلی) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ثابت int64_t n_kv = gguf_get_n_kv(ctx);

    لكل (int64_t i = 0; i < n_kv; ++i) {
        إذا (strcmp(کلی, gguf_get_کلی(ctx, i)) == 0) {
            keyfound = i;
            اكسر;
        }
    }

    أرجع keyfound;
}

ثابت حرف * gguf_get_کلی(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    أرجع ctx->kv[کلی_id].get_کلی().c_str();
}

تعداد gguf_type gguf_get_kv_type(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    أرجع ctx->kv[کلی_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[کلی_id].get_type();
}

تعداد gguf_type gguf_get_arr_type(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].is_array);
    أرجع ctx->kv[کلی_id].get_type();
}

ثابت فراغ * gguf_get_arr_معلومات(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_type() != GGUF_TYPE_STRING);
    أرجع ctx->kv[کلی_id].معلومات.معلومات();
}

ثابت حرف * gguf_get_arr_str(ثابت بنية gguf_context * ctx, int64_t کلی_id, size_t i) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_type() == GGUF_TYPE_STRING);
    أرجع ctx->kv[کلی_id].معلومات_string[i].c_str();
}

size_t gguf_get_arr_n(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));

    إذا (ctx->kv[کلی_id].type == GGUF_TYPE_STRING) {
        أرجع ctx->kv[کلی_id].معلومات_string.size();
    }

    ثابت size_t type_اندازه = gguf_type_اندازه(ctx->kv[کلی_id].type);
    GGML_ASSERT(ctx->kv[کلی_id].معلومات.size() % type_اندازه == 0);
    أرجع ctx->kv[کلی_id].معلومات.size() / type_اندازه;
}

uint8_t gguf_get_val_u8(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<int32_t>();
}

عائم gguf_get_val_f32(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<عائم>();
}

uint64_t gguf_get_val_u64(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<int64_t>();
}

مزدوج gguf_get_val_f64(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<مزدوج>();
}

منطقي gguf_get_val_bool(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<منطقي>();
}

ثابت حرف * gguf_get_val_str(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    أرجع ctx->kv[کلی_id].get_val<std::string>().c_str();
}

ثابت فراغ * gguf_get_val_معلومات(ثابت بنية gguf_context * ctx, int64_t کلی_id) {
    GGML_ASSERT(کلی_id >= 0 && کلی_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[کلی_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[کلی_id].get_type() != GGUF_TYPE_STRING);
    أرجع ctx->kv[کلی_id].معلومات.معلومات();
}

int64_t gguf_get_n_tensors(ثابت بنية gguf_context * ctx) {
    أرجع ctx->info.size();
}

int64_t gguf_پیدا_کړه_tensor(ثابت بنية gguf_context * ctx, ثابت حرف * name) {
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

size_t gguf_get_tensor_دننه_واټن(ثابت بنية gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    أرجع ctx->info[tensor_id].دننه_واټن;
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

int64_t gguf_remove_کلی(بنية gguf_context * ctx, ثابت حرف * کلی) {
    ثابت int64_t کلی_id = gguf_پیدا_کړه_کلی(ctx, کلی);
    إذا (کلی_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + کلی_id);
    }
    أرجع کلی_id;
}

کينډۍ<ډول_نوم T>
ساكن فراغ gguf_check_reserved_کلیګانې(ثابت std::string & کلی, ثابت T val) {
    إذا (کلی == GGUF_کلی_GENERAL_ALIGNMENT) {
        إذا constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_کلی_GENERAL_ALIGNMENT " must be power of 2");
        } وإلا {
            GGML_ناکارول_شوی(val);
            GGML_ABORT(GGUF_کلی_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

فراغ gguf_سېټ_val_u8(بنية gguf_context * ctx, ثابت حرف * کلی, uint8_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_i8(بنية gguf_context * ctx, ثابت حرف * کلی, int8_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_u16(بنية gguf_context * ctx, ثابت حرف * کلی, uint16_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_i16(بنية gguf_context * ctx, ثابت حرف * کلی, int16_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_u32(بنية gguf_context * ctx, ثابت حرف * کلی, uint32_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_i32(بنية gguf_context * ctx, ثابت حرف * کلی, int32_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_f32(بنية gguf_context * ctx, ثابت حرف * کلی, عائم val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_u64(بنية gguf_context * ctx, ثابت حرف * کلی, uint64_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_i64(بنية gguf_context * ctx, ثابت حرف * کلی, int64_t val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_f64(بنية gguf_context * ctx, ثابت حرف * کلی, مزدوج val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_bool(بنية gguf_context * ctx, ثابت حرف * کلی, منطقي val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, val);
}

فراغ gguf_سېټ_val_str(بنية gguf_context * ctx, ثابت حرف * کلی, ثابت حرف * val) {
    gguf_check_reserved_کلیګانې(کلی, val);
    gguf_remove_کلی(ctx, کلی);
    ctx->kv.emplace_back(کلی, std::string(val));
}

فراغ gguf_سېټ_arr_معلومات(بنية gguf_context * ctx, ثابت حرف * کلی, تعداد gguf_type type, ثابت فراغ * معلومات, size_t n) {
    gguf_check_reserved_کلیګانې(کلی, معلومات);
    gguf_remove_کلی(ctx, کلی);

    ثابت size_t nbytes = n*gguf_type_اندازه(type);
    std::vector<int8_t> tmp(nbytes);
    إذا (!tmp.تشه()) {
        حافظه_نقل(tmp.معلومات(), معلومات, nbytes);
    }
    ctx->kv.emplace_back(کلی, tmp);
    ctx->kv.back().cast(type);
}

فراغ gguf_سېټ_arr_str(بنية gguf_context * ctx, ثابت حرف * کلی, ثابت حرف ** معلومات, size_t n) {
    gguf_check_reserved_کلیګانې(کلی, معلومات);
    gguf_remove_کلی(ctx, کلی);

    std::vector<std::string> tmp(n);
    لكل (size_t i = 0; i < n; ++i) {
        tmp[i] = معلومات[i];
    }
    ctx->kv.emplace_back(کلی, tmp);
}

// set or add KV pairs from another context
فراغ gguf_سېټ_kv(بنية gguf_context * ctx, ثابت بنية gguf_context * src) {
    ثابت int64_t n_kv = gguf_get_n_kv(src);
    لكل (int64_t i = 0; i < n_kv; ++i) {
        ثابت بنية gguf_kv & kv = src->kv[i];

        إذا (!kv.is_array) {
            بدل (kv.get_type()) {
                حالة GGUF_TYPE_UINT8:   gguf_سېټ_val_u8  (ctx, kv.get_کلی().c_str(), kv.get_val<uint8_t>());             اكسر;
                حالة GGUF_TYPE_INT8:    gguf_سېټ_val_i8  (ctx, kv.get_کلی().c_str(), kv.get_val<int8_t>());              اكسر;
                حالة GGUF_TYPE_UINT16:  gguf_سېټ_val_u16 (ctx, kv.get_کلی().c_str(), kv.get_val<uint16_t>());            اكسر;
                حالة GGUF_TYPE_INT16:   gguf_سېټ_val_i16 (ctx, kv.get_کلی().c_str(), kv.get_val<int16_t>());             اكسر;
                حالة GGUF_TYPE_UINT32:  gguf_سېټ_val_u32 (ctx, kv.get_کلی().c_str(), kv.get_val<uint32_t>());            اكسر;
                حالة GGUF_TYPE_INT32:   gguf_سېټ_val_i32 (ctx, kv.get_کلی().c_str(), kv.get_val<int32_t>());             اكسر;
                حالة GGUF_TYPE_FLOAT32: gguf_سېټ_val_f32 (ctx, kv.get_کلی().c_str(), kv.get_val<عائم>());               اكسر;
                حالة GGUF_TYPE_UINT64:  gguf_سېټ_val_u64 (ctx, kv.get_کلی().c_str(), kv.get_val<uint64_t>());            اكسر;
                حالة GGUF_TYPE_INT64:   gguf_سېټ_val_i64 (ctx, kv.get_کلی().c_str(), kv.get_val<int64_t>());             اكسر;
                حالة GGUF_TYPE_FLOAT64: gguf_سېټ_val_f64 (ctx, kv.get_کلی().c_str(), kv.get_val<مزدوج>());              اكسر;
                حالة GGUF_TYPE_BOOL:    gguf_سېټ_val_bool(ctx, kv.get_کلی().c_str(), kv.get_val<منطقي>());                اكسر;
                حالة GGUF_TYPE_STRING:  gguf_سېټ_val_str (ctx, kv.get_کلی().c_str(), kv.get_val<std::string>().c_str()); اكسر;
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
                gguf_سېټ_arr_معلومات(ctx, kv.get_کلی().c_str(), kv.get_type(), kv.معلومات.معلومات(), ne);
            } اكسر;
            حالة GGUF_TYPE_STRING: {
                std::vector<ثابت حرف *> tmp(ne);
                لكل (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.معلومات_string[j].c_str();
                }
                gguf_سېټ_arr_str(ctx, kv.get_کلی().c_str(), tmp.معلومات(), ne);
            } اكسر;
            حالة GGUF_TYPE_ARRAY:
            افتراضي: GGML_ABORT("invalid type");
        }
    }
}

فراغ gguf_جمع_کړه_tensor(
             بنية gguf_context * ctx,
        ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    إذا (gguf_پیدا_کړه_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    بنية gguf_tensor_info ti;
    ti.t = *tensor;
    ti.دننه_واټن = ctx->info.تشه() ? 0 :
        ctx->info.back().دننه_واټن + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

فراغ gguf_سېټ_tensor_type(بنية gguf_context * ctx, ثابت حرف * name, تعداد ggml_type type) {
    ثابت int64_t tensor_id = gguf_پیدا_کړه_tensor(ctx, name);
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
        ctx->info[i].دننه_واټن = ctx->info[i - 1].دننه_واټن + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

فراغ gguf_سېټ_tensor_معلومات(بنية gguf_context * ctx, ثابت حرف * name, ثابت فراغ * معلومات) {
    ثابت int64_t tensor_id = gguf_پیدا_کړه_tensor(ctx, name);
    إذا (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.معلومات = (فراغ *)(uintptr_t)معلومات; // double cast suppresses warning about casting away const
}

بنية gguf_writer_بنسټ {
    size_t written_بایټونه {0u};

    مجازي ~gguf_writer_بنسټ(فراغ) = افتراضي;

    // we bet on devirtualization
    مجازي فراغ اكتب(int8_t val) = 0;
    مجازي فراغ اكتب(ثابت std::vector<int8_t> & val) = 0;
    مجازي فراغ write_tensor_معلومات(ثابت بنية gguf_tensor_info & info, size_t دننه_واټن_معلومات, size_t alignment) = 0;

    کينډۍ <ډول_نوم T>
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
            ثابت uint64_t n = val.اوږدتیا();
            اكتب(n);
        }
        لكل (size_t i = 0; i < val.اوږدتیا(); ++i) {
            اكتب((val.معلومات())[i]);
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

        اكتب(kv.get_کلی());

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
                اكتب(kv.معلومات);
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
        اكتب(info.دننه_واټن);
    }

    فراغ pad(ثابت size_t alignment) {
        طالما (written_بایټونه % alignment != 0) {
            ثابت int8_t zero = 0;
            اكتب(zero);
        }
    }
};

// vector buffer based writer
بنية gguf_writer_buf final : عام gguf_writer_بنسټ {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    کارول gguf_writer_بنسټ::اكتب;

    فراغ اكتب(ثابت int8_t val) override {
        buf.push_back(val);
        written_بایټونه++;
    }

    فراغ اكتب(ثابت std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_بایټونه += val.size();
    }

    فراغ write_tensor_معلومات(ثابت بنية gguf_tensor_info & info, ثابت size_t دننه_واټن_معلومات, ثابت size_t alignment) override {
        GGML_ASSERT(buf.size() - دننه_واټن_معلومات == info.دننه_واټن);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t دننه_واټن = buf.size();
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(دننه_واټن + nbytes);
        إذا (info.t.بفر) {
            ggml_backend_tensor_get(&info.t, buf.معلومات() + دننه_واټن, 0, nbytes);
        } وإلا {
            GGML_ASSERT(info.t.معلومات);
            حافظه_نقل(buf.معلومات() + دننه_واټن, info.t.معلومات, nbytes);
        }
        written_بایټونه += nbytes;

        pad(alignment);
    }
};

// file based writer
بنية gguf_writer_file final : عام gguf_writer_بنسټ {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    کارول gguf_writer_بنسټ::اكتب;

    فراغ اكتب(ثابت int8_t val) override {
        ثابت خپلکاری real_val = static_cast<uint8_t>(val);
        ثابت خپلکاری ret = fputc(real_val, file);
        written_بایټونه++;
        إذا (ret != real_val) {
            ارم std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((صحيح)real_val) + "'");
        }
    }

    فراغ اكتب(ثابت std::vector<int8_t> & val) override {
        ثابت خپلکاری ret = fwrite(val.معلومات(), 1, val.size(), file);
        written_بایټونه += val.size();
        إذا (ret != val.size()) {
            ارم std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    فراغ write_tensor_معلومات(ثابت بنية gguf_tensor_info & info, ثابت size_t دننه_واټن_معلومات, ثابت size_t alignment) override {
        GGML_ASSERT(written_بایټونه - دننه_واټن_معلومات == info.دننه_واټن);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ثابت size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        إذا (info.t.بفر) {
            ggml_backend_tensor_get(&info.t, buf.معلومات(), 0, nbytes);
        } وإلا {
            GGML_ASSERT(info.t.معلومات);
            حافظه_نقل(buf.معلومات(), info.t.معلومات, nbytes);
        }
        اكتب(buf);

        pad(alignment);
    }
};

کينډۍ <ډول_نوم writer_t>
ساكن فراغ gguf_write_بهر(ثابت بنية gguf_context * ctx, writer_t & gw, منطقي only_meta) {
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

    ثابت size_t دننه_واټن_معلومات = gw.written_بایټونه;

    // write tensor data
    لكل (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_معلومات(ctx->info[i], دننه_واټن_معلومات, ctx->alignment);
    }
}

فراغ gguf_write_منزل_buf(ثابت بنية gguf_context * ctx, std::vector<int8_t> & buf, منطقي only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_بهر(ctx, gw, only_meta);
}

منطقي gguf_write_منزل_file_ptr(ثابت بنية gguf_context * ctx, FILE * file, منطقي only_meta) {
    GGML_ASSERT(file);

    حاول {
        gguf_writer_file gw(file);
        gguf_write_بهر(ctx, gw, only_meta);
    } امسك (ثابت std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        أرجع خطأ;
    }
    أرجع ریښتیا;
}

منطقي gguf_write_منزل_file(ثابت بنية gguf_context * ctx, ثابت حرف * fname, منطقي only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    إذا (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        أرجع خطأ;
    }

    ثابت منطقي success = gguf_write_منزل_file_ptr(ctx, file, only_meta);
    إذا (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    أرجع success;
}

size_t gguf_get_meta_اندازه(ثابت بنية gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_منزل_buf(ctx, buf, /*only_meta =*/ ریښتیا);
    أرجع buf.size();
}

فراغ gguf_get_meta_معلومات(ثابت بنية gguf_context * ctx, فراغ * معلومات) {
    std::vector<int8_t> buf;
    gguf_write_منزل_buf(ctx, buf, /*only_meta =*/ ریښتیا);
    حافظه_نقل(معلومات, buf.معلومات(), buf.size());
}
