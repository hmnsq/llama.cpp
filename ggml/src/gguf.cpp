#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "gguf.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <cerrno>
#अन्तर्भाव <map>
#अन्तर्भाव <new>
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#अन्यथा
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

ढाँचा <typename T>
संरचना type_to_gguf_type;

ढाँचा <>
संरचना type_to_gguf_type<uint8_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT8;
};

ढाँचा <>
संरचना type_to_gguf_type<int8_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT8;
};

ढाँचा <>
संरचना type_to_gguf_type<uint16_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT16;
};

ढाँचा <>
संरचना type_to_gguf_type<int16_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT16;
};

ढाँचा <>
संरचना type_to_gguf_type<uint32_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT32;
};

ढाँचा <>
संरचना type_to_gguf_type<int32_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT32;
};

ढाँचा <>
संरचना type_to_gguf_type<भिन्न> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_FLOAT32;
};

ढाँचा <>
संरचना type_to_gguf_type<तार्किक> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_BOOL;
};

ढाँचा <>
संरचना type_to_gguf_type<std::string> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_STRING;
};

ढाँचा <>
संरचना type_to_gguf_type<uint64_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT64;
};

ढाँचा <>
संरचना type_to_gguf_type<int64_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT64;
};

ढाँचा <>
संरचना type_to_gguf_type<द्विगुण> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_FLOAT64;
};

स्थिर अचर std::map<gguf_type, माप_प्रकार> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   परिमाण(uint8_t)},
    {GGUF_TYPE_INT8,    परिमाण(int8_t)},
    {GGUF_TYPE_UINT16,  परिमाण(uint16_t)},
    {GGUF_TYPE_INT16,   परिमाण(int16_t)},
    {GGUF_TYPE_UINT32,  परिमाण(uint32_t)},
    {GGUF_TYPE_INT32,   परिमाण(int32_t)},
    {GGUF_TYPE_FLOAT32, परिमाण(भिन्न)},
    {GGUF_TYPE_BOOL,    परिमाण(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  परिमाण(uint64_t)},
    {GGUF_TYPE_INT64,   परिमाण(int64_t)},
    {GGUF_TYPE_FLOAT64, परिमाण(द्विगुण)},
};
static_assert(GGUF_TYPE_गन्ती == 13, "GGUF_TYPE_COUNT != 13");

स्थिर अचर std::map<gguf_type, अचर वर्ण *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_गन्ती == 13, "GGUF_TYPE_COUNT != 13");

माप_प्रकार gguf_type_size(गणना gguf_type type) {
    स्वतः it = GGUF_TYPE_SIZE.find(type);
    फर्काऊ it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

संरचना gguf_kv {
    std::string साँचो;

    तार्किक is_array;
    गणना gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    ढाँचा <typename T>
    gguf_kv(अचर std::string & साँचो, अचर T value)
            : साँचो(साँचो), is_array(असत्य), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!साँचो.empty());
        data.resize(परिमाण(T));
        स_नकल(data.data(), &value, परिमाण(T));
    }

    ढाँचा <typename T>
    gguf_kv(अचर std::string & साँचो, अचर std::vector<T> & value)
            : साँचो(साँचो), is_array(सत्य), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!साँचो.empty());
        data.resize(value.size()*परिमाण(T));
        लागि (माप_प्रकार i = 0; i < value.size(); ++i) {
            अचर T tmp = value[i];
            स_नकल(data.data() + i*परिमाण(T), &tmp, परिमाण(T));
        }
    }

    gguf_kv(अचर std::string & साँचो, अचर std::string & value)
            : साँचो(साँचो), is_array(असत्य), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!साँचो.empty());
        data_string.push_back(value);
    }

    gguf_kv(अचर std::string & साँचो, अचर std::vector<std::string> & value)
            : साँचो(साँचो), is_array(सत्य), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!साँचो.empty());
        data_string = value;
    }

    अचर std::string & get_साँचो() अचर {
        फर्काऊ साँचो;
    }

    अचर गणना gguf_type & get_type() अचर {
        फर्काऊ type;
    }

    माप_प्रकार get_ne() अचर {
        यदि (type == GGUF_TYPE_STRING) {
            अचर माप_प्रकार ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            फर्काऊ ne;
        }
        अचर माप_प्रकार type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        अचर माप_प्रकार ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        फर्काऊ ne;
    }

    ढाँचा <typename T>
    अचर T & get_val(अचर माप_प्रकार i = 0) अचर {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        यदि constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            फर्काऊ data_string[i];
        }
        अचर माप_प्रकार type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        फर्काऊ reinterpret_cast<अचर T *>(data.data())[i];
    }

    शून्य cast(अचर गणना gguf_type new_type) {
        अचर माप_प्रकार new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

संरचना gguf_tensor_info {
    संरचना ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

संरचना gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<संरचना gguf_kv> kv;
    std::vector<संरचना gguf_tensor_info> info;

    माप_प्रकार alignment = GGUF_DEFAULT_ALIGNMENT;
    माप_प्रकार offset    = 0; // offset of `data` from beginning of file
    माप_प्रकार size      = 0; // size of `data` in bytes

    शून्य * data = nullptr;
};

संरचना gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            शून्य * userdata,
            माप_प्रकार max_chunk_read,
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
    स्थिर uint64_t file_remain(खाता * file) {
        अचर int64_t cur = gguf_ftell(file);
        यदि (cur < 0) {
            फर्काऊ 0;
        }
        यदि (gguf_fseek(file, 0, अंत_से) != 0) {
            gguf_fseek(file, cur, शुरू_से);

            फर्काऊ 0;
        }
        अचर int64_t end = gguf_ftell(file);
        यदि (end < 0) {
            gguf_fseek(file, cur, शुरू_से);

            फर्काऊ 0;
        }
        gguf_fseek(file, cur, शुरू_से);
        फर्काऊ static_cast<uint64_t>(end - cur);
    }

    ढाँचा <typename T>
    तार्किक पठ(T & dst) अचर {
        अचर माप_प्रकार size = परिमाण(dst);
        यदि (size > nbytes_remain) {
            फर्काऊ असत्य;
        }
        फर्काऊ read_raw(&dst, size) == size;
    }

    ढाँचा <typename T>
    तार्किक पठ(std::vector<T> & dst, अचर माप_प्रकार n) अचर {
        यदि (n > GGUF_MAX_ARRAY_ELEMENTS) {
            फर्काऊ असत्य;
        }
        यदि constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            यदि (n > SIZE_MAX / परिमाण(uint64_t)) {
                फर्काऊ असत्य;
            }
            यदि (nbytes_remain < n * परिमाण(uint64_t)) {
                फर्काऊ असत्य;
            }
        } अन्यथा {
            यदि (n > SIZE_MAX / परिमाण(T)) {
                फर्काऊ असत्य;
            }
            यदि (nbytes_remain < n * परिमाण(T)) {
                फर्काऊ असत्य;
            }
        }
        dst.resize(n);
        लागि (माप_प्रकार i = 0; i < dst.size(); ++i) {
            यदि constexpr (std::is_same<T, तार्किक>::value) {
                तार्किक tmp;
                यदि (!पठ(tmp)) {
                    फर्काऊ असत्य;
                }
                dst[i] = tmp;
            } अन्यथा {
                यदि (!पठ(dst[i])) {
                    फर्काऊ असत्य;
                }
            }
        }
        फर्काऊ सत्य;
    }

    तार्किक पठ(तार्किक & dst) अचर {
        int8_t tmp = -1;
        यदि (!पठ(tmp)) {
            फर्काऊ असत्य;
        }
        dst = tmp != 0;
        फर्काऊ सत्य;
    }

    तार्किक पठ(गणना ggml_type & dst) अचर {
        int32_t tmp = -1;
        यदि (!पठ(tmp)) {
            फर्काऊ असत्य;
        }
        dst = ggml_type(tmp);
        फर्काऊ सत्य;
    }

    तार्किक पठ(गणना gguf_type & dst) अचर {
        int32_t tmp = -1;
        यदि (!पठ(tmp)) {
            फर्काऊ असत्य;
        }
        dst = gguf_type(tmp);
        फर्काऊ सत्य;
    }

    तार्किक पठ(std::string & dst) अचर {
        uint64_t size = 0;
        यदि (!पठ(size)) {
            फर्काऊ असत्य;
        }
        यदि (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            फर्काऊ असत्य;
        }
        यदि (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            फर्काऊ असत्य;
        }
        dst.resize(static_cast<माप_प्रकार>(size));
        फर्काऊ read_raw(dst.data(), static_cast<माप_प्रकार>(size)) == size;
    }

    तार्किक पठ(शून्य * dst, अचर माप_प्रकार size) अचर {
        यदि (size > nbytes_remain) {
            फर्काऊ असत्य;
        }
        फर्काऊ read_raw(dst, size) == size;
    }

    uint64_t tell() अचर {
        फर्काऊ data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() अचर {
        फर्काऊ start_offset;
    }

    तार्किक seek(uint64_t absolute_offset) अचर {
        अचर uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        यदि (absolute_offset > end_offset) {
            फर्काऊ असत्य;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        फर्काऊ सत्य;
    }

गुप्त:
    माप_प्रकार read_raw(शून्य * dst, माप_प्रकार size) अचर {
        यदि (callback == nullptr || size == 0) {
            फर्काऊ 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        माप_प्रकार total_nread = 0;
        तार्किक reached_eof = असत्य;

        जबसम्म (total_nread < size) {
            अचर माप_प्रकार chunk_size = std::min(max_chunk_read, size - total_nread);
            यदि (data_offset + total_nread < data_offset) {
                रोक;
            }
            अचर माप_प्रकार nread = callback(userdata, static_cast<शून्य *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            यदि (nread != chunk_size) {
                reached_eof = सत्य;
                रोक;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        यदि (reached_eof) {
            nbytes_remain = 0;
        }

        फर्काऊ total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    शून्य * userdata = nullptr;
    माप_प्रकार max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

संरचना gguf_context * gguf_init_empty(शून्य) {
    फर्काऊ नूतन gguf_context;
}

ढाँचा<typename T>
तार्किक gguf_read_emplace_helper(अचर संरचना gguf_reader & gr, std::vector<संरचना gguf_kv> & kv, अचर std::string & साँचो, अचर तार्किक is_array, अचर माप_प्रकार n) {
    यदि (is_array) {
        std::vector<T> value;
        यत्न {
            यदि (!gr.पठ(value, n)) {
                फर्काऊ असत्य;
            }
        } गृहाण (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, साँचो.c_str());
            फर्काऊ असत्य;
        } गृहाण (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, साँचो.c_str());
            फर्काऊ असत्य;
        }
        kv.emplace_back(साँचो, value);
    } अन्यथा {
        T value;
        यदि (!gr.पठ(value)) {
            फर्काऊ असत्य;
        }
        kv.emplace_back(साँचो, value);
    }
    फर्काऊ सत्य;
}

स्थिर संरचना gguf_context * gguf_init_from_reader(अचर संरचना gguf_reader & gr, संरचना gguf_init_params params) {
    संरचना gguf_context * ctx = नूतन gguf_context;

    तार्किक ok = सत्य;

    // file magic
    {
        std::vector<वर्ण> magic;
        ok = ok && gr.पठ(magic, 4);

        यदि (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            फर्काऊ nullptr;
        }

        लागि (uint32_t i = 0; i < magic.size(); i++) {
            यदि (magic[i] != GGUF_MAGIC[i]) {
                वर्ण c0 = है_छाप(magic[0]) ? magic[0] : '?';
                वर्ण c1 = है_छाप(magic[1]) ? magic[1] : '?';
                वर्ण c2 = है_छाप(magic[2]) ? magic[2] : '?';
                वर्ण c3 = है_छाप(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    यदि (ok && gr.पठ(ctx->version)) {
        यदि (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = असत्य;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        यदि (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = असत्य;
        }

        यदि (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = असत्य;
        }
        यदि (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = असत्य;
        }
    } अन्यथा {
        ok = असत्य;
    }

    यदि (ok && gr.पठ(n_tensors)) {
        static_assert(परिमाण(माप_प्रकार) <= 8 && परिमाण(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        यदि (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/परिमाण(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/परिमाण(gguf_tensor_info));
            ok = असत्य;
        }
    } अन्यथा {
        ok = असत्य;
    }

    यदि (ok && gr.पठ(n_kv)) {
        static_assert(परिमाण(माप_प्रकार) <= 8 && परिमाण(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        यदि (n_kv < 0 || n_kv > int64_t(SIZE_MAX/परिमाण(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/परिमाण(gguf_kv));
            ok = असत्य;
        }
    } अन्यथा {
        ok = असत्य;
    }

    यदि (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        फर्काऊ nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_साँचोहरू;

        लागि (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string साँचो;
            gguf_type   type     = gguf_type(-1);
            तार्किक        is_array = असत्य;
            uint64_t    n        = 1;

            यत्न {
                ok = ok && gr.पठ(साँचो);
            } गृहाण (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = असत्य;
            } गृहाण (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = असत्य;
            }
            यदि (ok && साँचो.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = असत्य;
            }
            यदि (ok && !seen_साँचोहरू.insert(साँचो).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, साँचो.c_str(), i);
                ok = असत्य;
            }
            यदि (!ok) {
                रोक;
            }

            ok = ok && gr.पठ(type);
            यदि (type == GGUF_TYPE_ARRAY) {
                is_array = सत्य;
                ok = ok && gr.पठ(type);
                ok = ok && gr.पठ(n);
            }
            यदि (!ok) {
                रोक;
            }

            विकल्प (type) {
                पक्ष GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<भिन्न>      (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<तार्किक>       (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<द्विगुण>     (gr, ctx->kv, साँचो, is_array, n); रोक;
                पक्ष GGUF_TYPE_ARRAY:
                स्वभाव:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, साँचो.c_str(), type);
                        ok = असत्य;
                    } रोक;
            }
        }

        यदि (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            फर्काऊ nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        अचर पूर्णाङ्क alignment_idx = gguf_find_साँचो(ctx, GGUF_साँचो_GENERAL_ALIGNMENT);
        यदि (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_साँचो_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            फर्काऊ nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        यदि (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            फर्काऊ nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    लागि (int64_t i = 0; ok && i < n_tensors; ++i) {
        संरचना gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            यत्न {
                ok = ok && gr.पठ(name);
            } गृहाण (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = असत्य;
            } गृहाण (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = असत्य;
            }
            यदि (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = असत्य;
                रोक;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            यदि (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = असत्य;
                रोक;
            }
        }
        यदि (!ok) {
            रोक;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.पठ(n_dims);
            यदि (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = असत्य;
                रोक;
            }
            लागि (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                यदि (j < n_dims) {
                    ok = ok && gr.पठ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                यदि (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = असत्य;
                    रोक;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            यदि (ok) {
                int64_t current_nelements = info.t.ne[0];
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                    यदि (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        यदि (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = असत्य;
                             रोक;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 यदि (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     रोक;
                 }
            }
        }
        यदि (!ok) {
            रोक;
        }

        // tensor type
        {
            ok = ok && gr.पठ(info.t.type);

            // check that tensor type is within defined range
            यदि (info.t.type < 0 || info.t.type >= GGML_TYPE_गन्ती) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_गन्ती);
                ok = असत्य;
                रोक;
            }
            अचर माप_प्रकार  type_size = ggml_type_size(info.t.type);
            अचर int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            यदि (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (पूर्णाङ्क) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = असत्य;
                रोक;
            }

            // check that the size of the tensor in bytes is representable
            यदि (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = असत्य;
                रोक;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            लागि (पूर्णाङ्क j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        यदि (!ok) {
            रोक;
        }

        // tensor data offset within buffer
        ok = ok && gr.पठ(info.offset);

        ctx->info.push_back(info);
    }

    यदि (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        फर्काऊ nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    यदि (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        फर्काऊ nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        लागि (माप_प्रकार i = 0; i < ctx->info.size(); ++i) {
            अचर gguf_tensor_info & ti = ctx->info[i];
            यदि (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }
            अचर माप_प्रकार nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            यदि (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }
            अचर माप_प्रकार padded_size = GGML_PAD(nbytes, ctx->alignment);
            यदि (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    यदि (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        माप_प्रकार mem_size = 0;
        यदि (params.no_alloc) {
            यदि (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }

            अचर माप_प्रकार overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } अन्यथा {
            यदि ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }

            अचर माप_प्रकार overhead = (n_tensors + 1) * ggml_tensor_overhead();

            यदि (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                फर्काऊ nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        संरचना ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        यदि (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            फर्काऊ nullptr;
        }

        संरचना ggml_context * ctx_data = *params.ctx;

        संरचना ggml_tensor * data = nullptr;

        यदि (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            यदि (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.पठ(data->data, ctx->size);

            यदि (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                फर्काऊ nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, सत्य);

        // create the tensors
        लागि (माप_प्रकार i = 0; i < ctx->info.size(); ++i) {
            अचर संरचना gguf_tensor_info & info = ctx->info[i];

            संरचना ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            यदि (!ok) {
                रोक;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            यदि (!params.no_alloc) {
                cur->data = (वर्ण *) data->data + info.offset;
            }
        }

        यदि (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            फर्काऊ nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    फर्काऊ ctx;
}

संरचना gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, शून्य * userdata, माप_प्रकार max_chunk_read, uint64_t max_expected_size, संरचना gguf_init_params params) {
    यदि (callback == nullptr) {
        फर्काऊ nullptr;
    }

    अचर संरचना gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    फर्काऊ gguf_init_from_reader(gr, params);
}

संरचना gguf_file_reader {
    खाता * file;
    uint64_t offset;
};

स्थिर माप_प्रकार gguf_file_reader_callback(शून्य * userdata, शून्य * output, uint64_t offset, माप_प्रकार len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    यदि (reader.offset != offset) {
        यदि (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), शुरू_से) != 0) {
            फर्काऊ 0;
        }

        reader.offset = offset;
    }

    अचर माप_प्रकार nread = ख_पढ़ो(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    फर्काऊ nread;
}

संरचना gguf_context * gguf_init_from_file_ptr(खाता * file, संरचना gguf_init_params params) {
    यदि (!file) {
        फर्काऊ nullptr;
    }

    अचर int64_t cur = gguf_ftell(file);
    यदि (cur < 0) {
        फर्काऊ nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    अचर संरचना gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    फर्काऊ gguf_init_from_reader(gr, params);
}

संरचना gguf_buffer_reader {
    अचर uint8_t * data;
    माप_प्रकार          size;
};

स्थिर माप_प्रकार gguf_buffer_reader_callback(शून्य * userdata, शून्य * output, uint64_t offset, माप_प्रकार len) {
    GGML_ASSERT(len > 0);

    अचर gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    यदि (offset > reader.size || len > reader.size - offset) {
        फर्काऊ 0;
    }

    अचर माप_प्रकार data_offset = static_cast<माप_प्रकार>(offset);
    अचर माप_प्रकार nread = std::min(len, reader.size - data_offset);
    स_नकल(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    फर्काऊ nread;
}

संरचना gguf_context * gguf_init_from_buffer(अचर शून्य * data, माप_प्रकार size, संरचना gguf_init_params params) {
    यदि (data == nullptr || size == 0) {
        फर्काऊ nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<अचर uint8_t *>(data),
        /*.size = */ size,
    };
    अचर संरचना gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    फर्काऊ gguf_init_from_reader(gr, params);
}

संरचना gguf_context * gguf_init_from_file(अचर वर्ण * fname, संरचना gguf_init_params params) {
    खाता * file = ggml_fopen(fname, "rb");

    यदि (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, म_त्रुटि(त्रुटि_सं));
        फर्काऊ nullptr;
    }

    संरचना gguf_context * result = gguf_init_from_file_ptr(file, params);
    ख_बंद(file);
    फर्काऊ result;
}

शून्य gguf_free(संरचना gguf_context * ctx) {
    यदि (ctx == nullptr) {
        फर्काऊ;
    }
    delete ctx;
}

अचर वर्ण * gguf_type_name(गणना gguf_type type) {
    स्वतः it = GGUF_TYPE_NAME.find(type);
    फर्काऊ it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(अचर संरचना gguf_context * ctx) {
    फर्काऊ ctx->version;
}

माप_प्रकार gguf_get_alignment(अचर संरचना gguf_context * ctx) {
    फर्काऊ ctx->alignment;
}

माप_प्रकार gguf_get_data_offset(अचर संरचना gguf_context * ctx) {
    फर्काऊ ctx->offset;
}

int64_t gguf_get_n_kv(अचर संरचना gguf_context * ctx) {
    फर्काऊ ctx->kv.size();
}

int64_t gguf_find_साँचो(अचर संरचना gguf_context * ctx, अचर वर्ण * साँचो) {
    // return -1 if key not found
    int64_t keyfound = -1;

    अचर int64_t n_kv = gguf_get_n_kv(ctx);

    लागि (int64_t i = 0; i < n_kv; ++i) {
        यदि (म_भेद(साँचो, gguf_get_साँचो(ctx, i)) == 0) {
            keyfound = i;
            रोक;
        }
    }

    फर्काऊ keyfound;
}

अचर वर्ण * gguf_get_साँचो(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    फर्काऊ ctx->kv[साँचो_id].get_साँचो().c_str();
}

गणना gguf_type gguf_get_kv_type(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    फर्काऊ ctx->kv[साँचो_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[साँचो_id].get_type();
}

गणना gguf_type gguf_get_arr_type(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].is_array);
    फर्काऊ ctx->kv[साँचो_id].get_type();
}

अचर शून्य * gguf_get_arr_data(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_type() != GGUF_TYPE_STRING);
    फर्काऊ ctx->kv[साँचो_id].data.data();
}

अचर वर्ण * gguf_get_arr_str(अचर संरचना gguf_context * ctx, int64_t साँचो_id, माप_प्रकार i) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_type() == GGUF_TYPE_STRING);
    फर्काऊ ctx->kv[साँचो_id].data_string[i].c_str();
}

माप_प्रकार gguf_get_arr_n(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));

    यदि (ctx->kv[साँचो_id].type == GGUF_TYPE_STRING) {
        फर्काऊ ctx->kv[साँचो_id].data_string.size();
    }

    अचर माप_प्रकार type_size = gguf_type_size(ctx->kv[साँचो_id].type);
    GGML_ASSERT(ctx->kv[साँचो_id].data.size() % type_size == 0);
    फर्काऊ ctx->kv[साँचो_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<int32_t>();
}

भिन्न gguf_get_val_f32(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<भिन्न>();
}

uint64_t gguf_get_val_u64(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<int64_t>();
}

द्विगुण gguf_get_val_f64(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<द्विगुण>();
}

तार्किक gguf_get_val_bool(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<तार्किक>();
}

अचर वर्ण * gguf_get_val_str(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    फर्काऊ ctx->kv[साँचो_id].get_val<std::string>().c_str();
}

अचर शून्य * gguf_get_val_data(अचर संरचना gguf_context * ctx, int64_t साँचो_id) {
    GGML_ASSERT(साँचो_id >= 0 && साँचो_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[साँचो_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[साँचो_id].get_type() != GGUF_TYPE_STRING);
    फर्काऊ ctx->kv[साँचो_id].data.data();
}

int64_t gguf_get_n_tensors(अचर संरचना gguf_context * ctx) {
    फर्काऊ ctx->info.size();
}

int64_t gguf_find_tensor(अचर संरचना gguf_context * ctx, अचर वर्ण * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    अचर int64_t n_tensors = gguf_get_n_tensors(ctx);

    लागि (int64_t i = 0; i < n_tensors; ++i) {
        यदि (म_भेद(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            रोक;
        }
    }

    फर्काऊ tensor_id;
}

माप_प्रकार gguf_get_tensor_offset(अचर संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    फर्काऊ ctx->info[tensor_id].offset;
}

अचर वर्ण * gguf_get_tensor_name(अचर संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    फर्काऊ ctx->info[tensor_id].t.name;
}

अचर int64_t * gguf_get_tensor_ne(अचर संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    फर्काऊ ctx->info[tensor_id].t.ne;
}

गणना ggml_type gguf_get_tensor_type(अचर संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    फर्काऊ ctx->info[tensor_id].t.type;
}

माप_प्रकार gguf_get_tensor_size(अचर संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    फर्काऊ ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_साँचो(संरचना gguf_context * ctx, अचर वर्ण * साँचो) {
    अचर int64_t साँचो_id = gguf_find_साँचो(ctx, साँचो);
    यदि (साँचो_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + साँचो_id);
    }
    फर्काऊ साँचो_id;
}

ढाँचा<typename T>
स्थिर शून्य gguf_check_reserved_साँचोहरू(अचर std::string & साँचो, अचर T val) {
    यदि (साँचो == GGUF_साँचो_GENERAL_ALIGNMENT) {
        यदि constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_साँचो_GENERAL_ALIGNMENT " must be power of 2");
        } अन्यथा {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_साँचो_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

शून्य gguf_set_val_u8(संरचना gguf_context * ctx, अचर वर्ण * साँचो, uint8_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_i8(संरचना gguf_context * ctx, अचर वर्ण * साँचो, int8_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_u16(संरचना gguf_context * ctx, अचर वर्ण * साँचो, uint16_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_i16(संरचना gguf_context * ctx, अचर वर्ण * साँचो, int16_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_u32(संरचना gguf_context * ctx, अचर वर्ण * साँचो, uint32_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_i32(संरचना gguf_context * ctx, अचर वर्ण * साँचो, int32_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_f32(संरचना gguf_context * ctx, अचर वर्ण * साँचो, भिन्न val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_u64(संरचना gguf_context * ctx, अचर वर्ण * साँचो, uint64_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_i64(संरचना gguf_context * ctx, अचर वर्ण * साँचो, int64_t val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_f64(संरचना gguf_context * ctx, अचर वर्ण * साँचो, द्विगुण val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_bool(संरचना gguf_context * ctx, अचर वर्ण * साँचो, तार्किक val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, val);
}

शून्य gguf_set_val_str(संरचना gguf_context * ctx, अचर वर्ण * साँचो, अचर वर्ण * val) {
    gguf_check_reserved_साँचोहरू(साँचो, val);
    gguf_remove_साँचो(ctx, साँचो);
    ctx->kv.emplace_back(साँचो, std::string(val));
}

शून्य gguf_set_arr_data(संरचना gguf_context * ctx, अचर वर्ण * साँचो, गणना gguf_type type, अचर शून्य * data, माप_प्रकार n) {
    gguf_check_reserved_साँचोहरू(साँचो, data);
    gguf_remove_साँचो(ctx, साँचो);

    अचर माप_प्रकार nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    यदि (!tmp.empty()) {
        स_नकल(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(साँचो, tmp);
    ctx->kv.back().cast(type);
}

शून्य gguf_set_arr_str(संरचना gguf_context * ctx, अचर वर्ण * साँचो, अचर वर्ण ** data, माप_प्रकार n) {
    gguf_check_reserved_साँचोहरू(साँचो, data);
    gguf_remove_साँचो(ctx, साँचो);

    std::vector<std::string> tmp(n);
    लागि (माप_प्रकार i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(साँचो, tmp);
}

// set or add KV pairs from another context
शून्य gguf_set_kv(संरचना gguf_context * ctx, अचर संरचना gguf_context * src) {
    अचर int64_t n_kv = gguf_get_n_kv(src);
    लागि (int64_t i = 0; i < n_kv; ++i) {
        अचर संरचना gguf_kv & kv = src->kv[i];

        यदि (!kv.is_array) {
            विकल्प (kv.get_type()) {
                पक्ष GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_साँचो().c_str(), kv.get_val<uint8_t>());             रोक;
                पक्ष GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_साँचो().c_str(), kv.get_val<int8_t>());              रोक;
                पक्ष GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_साँचो().c_str(), kv.get_val<uint16_t>());            रोक;
                पक्ष GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_साँचो().c_str(), kv.get_val<int16_t>());             रोक;
                पक्ष GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_साँचो().c_str(), kv.get_val<uint32_t>());            रोक;
                पक्ष GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_साँचो().c_str(), kv.get_val<int32_t>());             रोक;
                पक्ष GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_साँचो().c_str(), kv.get_val<भिन्न>());               रोक;
                पक्ष GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_साँचो().c_str(), kv.get_val<uint64_t>());            रोक;
                पक्ष GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_साँचो().c_str(), kv.get_val<int64_t>());             रोक;
                पक्ष GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_साँचो().c_str(), kv.get_val<द्विगुण>());              रोक;
                पक्ष GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_साँचो().c_str(), kv.get_val<तार्किक>());                रोक;
                पक्ष GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_साँचो().c_str(), kv.get_val<std::string>().c_str()); रोक;
                पक्ष GGUF_TYPE_ARRAY:
                स्वभाव: GGML_ABORT("invalid type");
            }
            जारी;
        }

        अचर माप_प्रकार ne = kv.get_ne();

        विकल्प (kv.get_type()) {
            पक्ष GGUF_TYPE_UINT8:
            पक्ष GGUF_TYPE_INT8:
            पक्ष GGUF_TYPE_UINT16:
            पक्ष GGUF_TYPE_INT16:
            पक्ष GGUF_TYPE_UINT32:
            पक्ष GGUF_TYPE_INT32:
            पक्ष GGUF_TYPE_FLOAT32:
            पक्ष GGUF_TYPE_UINT64:
            पक्ष GGUF_TYPE_INT64:
            पक्ष GGUF_TYPE_FLOAT64:
            पक्ष GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_साँचो().c_str(), kv.get_type(), kv.data.data(), ne);
            } रोक;
            पक्ष GGUF_TYPE_STRING: {
                std::vector<अचर वर्ण *> tmp(ne);
                लागि (माप_प्रकार j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_साँचो().c_str(), tmp.data(), ne);
            } रोक;
            पक्ष GGUF_TYPE_ARRAY:
            स्वभाव: GGML_ABORT("invalid type");
        }
    }
}

शून्य gguf_add_tensor(
             संरचना gguf_context * ctx,
        अचर संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    यदि (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    संरचना gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

शून्य gguf_set_tensor_type(संरचना gguf_context * ctx, अचर वर्ण * name, गणना ggml_type type) {
    अचर int64_t tensor_id = gguf_find_tensor(ctx, name);
    यदि (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    संरचना ggml_tensor * tensor = &ctx->info[tensor_id].t;
    अचर माप_प्रकार  type_size = ggml_type_size(type);
    अचर int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    लागि (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    अचर int64_t n_tensors = gguf_get_n_tensors(ctx);
    लागि (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

शून्य gguf_set_tensor_data(संरचना gguf_context * ctx, अचर वर्ण * name, अचर शून्य * data) {
    अचर int64_t tensor_id = gguf_find_tensor(ctx, name);
    यदि (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (शून्य *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

संरचना gguf_writer_base {
    माप_प्रकार written_bytes {0u};

    आभास ~gguf_writer_base(शून्य) = स्वभाव;

    // we bet on devirtualization
    आभास शून्य लिख(int8_t val) = 0;
    आभास शून्य लिख(अचर std::vector<int8_t> & val) = 0;
    आभास शून्य write_tensor_data(अचर संरचना gguf_tensor_info & info, माप_प्रकार offset_data, माप_प्रकार alignment) = 0;

    ढाँचा <typename T>
    शून्य लिख(अचर T & val) {
        लागि (माप_प्रकार i = 0; i < परिमाण(val); ++i) {
            लिख(reinterpret_cast<अचर int8_t *>(&val)[i]);
        }
    }

    शून्य लिख(अचर तार्किक & val) {
        अचर int8_t val8 = val ? 1 : 0;
        लिख(val8);
    }

    शून्य लिख(अचर std::string & val) {
        {
            अचर uint64_t n = val.length();
            लिख(n);
        }
        लागि (माप_प्रकार i = 0; i < val.length(); ++i) {
            लिख((val.data())[i]);
        }
    }

    शून्य लिख(अचर वर्ण * val) {
        लिख(std::string(val));
    }

    शून्य लिख(अचर गणना ggml_type & val) {
        लिख(int32_t(val));
    }

    शून्य लिख(अचर गणना gguf_type & val) {
        लिख(int32_t(val));
    }

    शून्य लिख(अचर संरचना gguf_kv & kv) {
        अचर uint64_t ne = kv.get_ne();

        लिख(kv.get_साँचो());

        यदि (kv.is_array) {
            लिख(GGUF_TYPE_ARRAY);
            लिख(kv.get_type());
            लिख(ne);
        } अन्यथा {
            लिख(kv.get_type());
        }

        विकल्प (kv.get_type()) {
            पक्ष GGUF_TYPE_UINT8:
            पक्ष GGUF_TYPE_INT8:
            पक्ष GGUF_TYPE_UINT16:
            पक्ष GGUF_TYPE_INT16:
            पक्ष GGUF_TYPE_UINT32:
            पक्ष GGUF_TYPE_INT32:
            पक्ष GGUF_TYPE_FLOAT32:
            पक्ष GGUF_TYPE_UINT64:
            पक्ष GGUF_TYPE_INT64:
            पक्ष GGUF_TYPE_FLOAT64: {
                लिख(kv.data);
            } रोक;
            पक्ष GGUF_TYPE_BOOL: {
                लागि (माप_प्रकार i = 0; i < ne; ++i) {
                    लिख(kv.get_val<तार्किक>(i));
                }
            } रोक;
            पक्ष GGUF_TYPE_STRING: {
                लागि (माप_प्रकार i = 0; i < ne; ++i) {
                    लिख(kv.get_val<std::string>(i));
                }
            } रोक;
            पक्ष GGUF_TYPE_ARRAY:
            स्वभाव: GGML_ABORT("invalid type");
        }
    }

    शून्य write_tensor_meta(अचर संरचना gguf_tensor_info & info) {
        लिख(info.t.name);

        अचर uint32_t n_dims = ggml_n_dims(&info.t);
        लिख(n_dims);

        लागि (uint32_t j = 0; j < n_dims; ++j) {
            लिख(info.t.ne[j]);
        }
        लिख(info.t.type);
        लिख(info.offset);
    }

    शून्य pad(अचर माप_प्रकार alignment) {
        जबसम्म (written_bytes % alignment != 0) {
            अचर int8_t zero = 0;
            लिख(zero);
        }
    }
};

// vector buffer based writer
संरचना gguf_writer_buf final : सार्वजनिक gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::लिख;

    शून्य लिख(अचर int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    शून्य लिख(अचर std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    शून्य write_tensor_data(अचर संरचना gguf_tensor_info & info, अचर माप_प्रकार offset_data, अचर माप_प्रकार alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        अचर माप_प्रकार offset = buf.size();
        अचर माप_प्रकार nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        यदि (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } अन्यथा {
            GGML_ASSERT(info.t.data);
            स_नकल(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
संरचना gguf_writer_file final : सार्वजनिक gguf_writer_base {
    खाता * file;

    gguf_writer_file(खाता* file) : file(file) {}

    using gguf_writer_base::लिख;

    शून्य लिख(अचर int8_t val) override {
        अचर स्वतः real_val = static_cast<uint8_t>(val);
        अचर स्वतः ret = ख_अक्षर_दो(real_val, file);
        written_bytes++;
        यदि (ret != real_val) {
            क्षिप std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((पूर्णाङ्क)real_val) + "'");
        }
    }

    शून्य लिख(अचर std::vector<int8_t> & val) override {
        अचर स्वतः ret = ख_डालो(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        यदि (ret != val.size()) {
            क्षिप std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    शून्य write_tensor_data(अचर संरचना gguf_tensor_info & info, अचर माप_प्रकार offset_data, अचर माप_प्रकार alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        अचर माप_प्रकार nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        यदि (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } अन्यथा {
            GGML_ASSERT(info.t.data);
            स_नकल(buf.data(), info.t.data, nbytes);
        }
        लिख(buf);

        pad(alignment);
    }
};

ढाँचा <typename writer_t>
स्थिर शून्य gguf_write_out(अचर संरचना gguf_context * ctx, writer_t & gw, तार्किक only_meta) {
    अचर int64_t n_kv      = gguf_get_n_kv(ctx);
    अचर int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.लिख(GGUF_MAGIC[0]);
    gw.लिख(GGUF_MAGIC[1]);
    gw.लिख(GGUF_MAGIC[2]);
    gw.लिख(GGUF_MAGIC[3]);
    gw.लिख(ctx->version);
    gw.लिख(n_tensors);
    gw.लिख(n_kv);

    // write key-value pairs
    लागि (int64_t i = 0; i < n_kv; ++i) {
        gw.लिख(ctx->kv[i]);
    }

    // write tensor info
    लागि (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    यदि (only_meta) {
        फर्काऊ;
    }

    अचर माप_प्रकार offset_data = gw.written_bytes;

    // write tensor data
    लागि (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

शून्य gguf_write_to_buf(अचर संरचना gguf_context * ctx, std::vector<int8_t> & buf, तार्किक only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

तार्किक gguf_write_to_file_ptr(अचर संरचना gguf_context * ctx, खाता * file, तार्किक only_meta) {
    GGML_ASSERT(file);

    यत्न {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } गृहाण (अचर std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        फर्काऊ असत्य;
    }
    फर्काऊ सत्य;
}

तार्किक gguf_write_to_file(अचर संरचना gguf_context * ctx, अचर वर्ण * fname, तार्किक only_meta) {
    खाता * file = ggml_fopen(fname, "wb");

    यदि (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        फर्काऊ असत्य;
    }

    अचर तार्किक success = gguf_write_to_file_ptr(ctx, file, only_meta);
    यदि (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    ख_बंद(file);
    फर्काऊ success;
}

माप_प्रकार gguf_get_meta_size(अचर संरचना gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ सत्य);
    फर्काऊ buf.size();
}

शून्य gguf_get_meta_data(अचर संरचना gguf_context * ctx, शून्य * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ सत्य);
    स_नकल(data, buf.data(), buf.size());
}
