#समावेश "ggml.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "gguf.h"

#समावेश <cinttypes>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <cstdio>
#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <cerrno>
#समावेश <map>
#समावेश <new>
#समावेश <stdexcept>
#समावेश <string>
#समावेश <unordered_set>
#समावेश <vector>

#घोषणा GGUF_MAX_STRING_लंबाई  (1024*1024*1024)
#घोषणा GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#अगर_घोषित _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#अन्यथा
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#पूर्ण_अगर

ढाँचा <typename T>
काष्ठा type_to_gguf_type;

ढाँचा <>
काष्ठा type_to_gguf_type<uint8_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_UINT8;
};

ढाँचा <>
काष्ठा type_to_gguf_type<int8_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_INT8;
};

ढाँचा <>
काष्ठा type_to_gguf_type<uint16_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_UINT16;
};

ढाँचा <>
काष्ठा type_to_gguf_type<int16_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_INT16;
};

ढाँचा <>
काष्ठा type_to_gguf_type<uint32_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_UINT32;
};

ढाँचा <>
काष्ठा type_to_gguf_type<int32_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_INT32;
};

ढाँचा <>
काष्ठा type_to_gguf_type<भग्न> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_FLOAT32;
};

ढाँचा <>
काष्ठा type_to_gguf_type<bool> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_BOOL;
};

ढाँचा <>
काष्ठा type_to_gguf_type<std::string> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_STRING;
};

ढाँचा <>
काष्ठा type_to_gguf_type<uint64_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_UINT64;
};

ढाँचा <>
काष्ठा type_to_gguf_type<int64_t> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_INT64;
};

ढाँचा <>
काष्ठा type_to_gguf_type<द्विगुन> {
    अटल constexpr क्रमागत gguf_type value = GGUF_TYPE_FLOAT64;
};

अटल स्थिर std::map<gguf_type, माप_प्रकार> GGUF_TYPE_आकार = {
    {GGUF_TYPE_UINT8,   माप(uint8_t)},
    {GGUF_TYPE_INT8,    माप(int8_t)},
    {GGUF_TYPE_UINT16,  माप(uint16_t)},
    {GGUF_TYPE_INT16,   माप(int16_t)},
    {GGUF_TYPE_UINT32,  माप(uint32_t)},
    {GGUF_TYPE_INT32,   माप(int32_t)},
    {GGUF_TYPE_FLOAT32, माप(भग्न)},
    {GGUF_TYPE_BOOL,    माप(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  माप(uint64_t)},
    {GGUF_TYPE_INT64,   माप(int64_t)},
    {GGUF_TYPE_FLOAT64, माप(द्विगुन)},
};
static_assert(GGUF_TYPE_गिनती == 13, "GGUF_TYPE_COUNT != 13");

अटल स्थिर std::map<gguf_type, स्थिर अक्षर *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_गिनती == 13, "GGUF_TYPE_COUNT != 13");

माप_प्रकार gguf_type_आकार(क्रमागत gguf_type type) {
    स्वतः it = GGUF_TYPE_आकार.find(type);
    वापस it == GGUF_TYPE_आकार.end() ? 0 : it->second;
}

काष्ठा gguf_kv {
    std::string कुंजी;

    bool is_array;
    क्रमागत gguf_type type;

    std::vector<int8_t>      तथ्य;
    std::vector<std::string> तथ्य_string;

    ढाँचा <typename T>
    gguf_kv(स्थिर std::string & कुंजी, स्थिर T value)
            : कुंजी(कुंजी), is_array(false), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!कुंजी.empty());
        तथ्य.resize(माप(T));
        स_नकल(तथ्य.तथ्य(), &value, माप(T));
    }

    ढाँचा <typename T>
    gguf_kv(स्थिर std::string & कुंजी, स्थिर std::vector<T> & value)
            : कुंजी(कुंजी), is_array(true), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!कुंजी.empty());
        तथ्य.resize(value.size()*माप(T));
        क्रम (माप_प्रकार i = 0; i < value.size(); ++i) {
            स्थिर T tmp = value[i];
            स_नकल(तथ्य.तथ्य() + i*माप(T), &tmp, माप(T));
        }
    }

    gguf_kv(स्थिर std::string & कुंजी, स्थिर std::string & value)
            : कुंजी(कुंजी), is_array(false), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!कुंजी.empty());
        तथ्य_string.push_back(value);
    }

    gguf_kv(स्थिर std::string & कुंजी, स्थिर std::vector<std::string> & value)
            : कुंजी(कुंजी), is_array(true), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!कुंजी.empty());
        तथ्य_string = value;
    }

    स्थिर std::string & get_कुंजी() स्थिर {
        वापस कुंजी;
    }

    स्थिर क्रमागत gguf_type & get_type() स्थिर {
        वापस type;
    }

    माप_प्रकार get_ne() स्थिर {
        अगर (type == GGUF_TYPE_STRING) {
            स्थिर माप_प्रकार ne = तथ्य_string.size();
            GGML_ASSERT(is_array || ne == 1);
            वापस ne;
        }
        स्थिर माप_प्रकार type_आकार = gguf_type_आकार(type);
        GGML_ASSERT(तथ्य.size() % type_आकार == 0);
        स्थिर माप_प्रकार ne = तथ्य.size() / type_आकार;
        GGML_ASSERT(is_array || ne == 1);
        वापस ne;
    }

    ढाँचा <typename T>
    स्थिर T & get_val(स्थिर माप_प्रकार i = 0) स्थिर {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        अगर constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(तथ्य_string.size() >= i+1);
            वापस तथ्य_string[i];
        }
        स्थिर माप_प्रकार type_आकार = gguf_type_आकार(type);
        GGML_ASSERT(तथ्य.size() % type_आकार == 0);
        GGML_ASSERT(तथ्य.size() >= (i+1)*type_आकार);
        वापस reinterpret_cast<स्थिर T *>(तथ्य.तथ्य())[i];
    }

    व्योम cast(स्थिर क्रमागत gguf_type new_type) {
        स्थिर माप_प्रकार new_type_आकार = gguf_type_आकार(new_type);
        GGML_ASSERT(तथ्य.size() % new_type_आकार == 0);
        type = new_type;
    }
};

काष्ठा gguf_tensor_info {
    काष्ठा ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

काष्ठा gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<काष्ठा gguf_kv> kv;
    std::vector<काष्ठा gguf_tensor_info> info;

    माप_प्रकार alignment = GGUF_DEFAULT_ALIGNMENT;
    माप_प्रकार offset    = 0; // offset of `data` from beginning of file
    माप_प्रकार size      = 0; // size of `data` in bytes

    व्योम * तथ्य = nullptr;
};

काष्ठा gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            व्योम * userdata,
            माप_प्रकार max_chunk_read,
            uint64_t तथ्य_offset = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          start_offset(तथ्य_offset),
          तथ्य_offset(तथ्य_offset),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    अटल uint64_t file_remain(खाता * file) {
        स्थिर int64_t cur = gguf_ftell(file);
        अगर (cur < 0) {
            वापस 0;
        }
        अगर (gguf_fseek(file, 0, अंत_से) != 0) {
            gguf_fseek(file, cur, शुरू_से);

            वापस 0;
        }
        स्थिर int64_t end = gguf_ftell(file);
        अगर (end < 0) {
            gguf_fseek(file, cur, शुरू_से);

            वापस 0;
        }
        gguf_fseek(file, cur, शुरू_से);
        वापस static_cast<uint64_t>(end - cur);
    }

    ढाँचा <typename T>
    bool पढ़ो(T & dst) स्थिर {
        स्थिर माप_प्रकार size = माप(dst);
        अगर (size > nbytes_remain) {
            वापस false;
        }
        वापस read_raw(&dst, size) == size;
    }

    ढाँचा <typename T>
    bool पढ़ो(std::vector<T> & dst, स्थिर माप_प्रकार n) स्थिर {
        अगर (n > GGUF_MAX_ARRAY_ELEMENTS) {
            वापस false;
        }
        अगर constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            अगर (n > SIZE_MAX / माप(uint64_t)) {
                वापस false;
            }
            अगर (nbytes_remain < n * माप(uint64_t)) {
                वापस false;
            }
        } अन्यथा {
            अगर (n > SIZE_MAX / माप(T)) {
                वापस false;
            }
            अगर (nbytes_remain < n * माप(T)) {
                वापस false;
            }
        }
        dst.resize(n);
        क्रम (माप_प्रकार i = 0; i < dst.size(); ++i) {
            अगर constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                अगर (!पढ़ो(tmp)) {
                    वापस false;
                }
                dst[i] = tmp;
            } अन्यथा {
                अगर (!पढ़ो(dst[i])) {
                    वापस false;
                }
            }
        }
        वापस true;
    }

    bool पढ़ो(bool & dst) स्थिर {
        int8_t tmp = -1;
        अगर (!पढ़ो(tmp)) {
            वापस false;
        }
        dst = tmp != 0;
        वापस true;
    }

    bool पढ़ो(क्रमागत ggml_type & dst) स्थिर {
        int32_t tmp = -1;
        अगर (!पढ़ो(tmp)) {
            वापस false;
        }
        dst = ggml_type(tmp);
        वापस true;
    }

    bool पढ़ो(क्रमागत gguf_type & dst) स्थिर {
        int32_t tmp = -1;
        अगर (!पढ़ो(tmp)) {
            वापस false;
        }
        dst = gguf_type(tmp);
        वापस true;
    }

    bool पढ़ो(std::string & dst) स्थिर {
        uint64_t size = 0;
        अगर (!पढ़ो(size)) {
            वापस false;
        }
        अगर (size > GGUF_MAX_STRING_लंबाई) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_लंबाई);
            वापस false;
        }
        अगर (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            वापस false;
        }
        dst.resize(static_cast<माप_प्रकार>(size));
        वापस read_raw(dst.तथ्य(), static_cast<माप_प्रकार>(size)) == size;
    }

    bool पढ़ो(व्योम * dst, स्थिर माप_प्रकार size) स्थिर {
        अगर (size > nbytes_remain) {
            वापस false;
        }
        वापस read_raw(dst, size) == size;
    }

    uint64_t tell() स्थिर {
        वापस तथ्य_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() स्थिर {
        वापस start_offset;
    }

    bool seek(uint64_t absolute_offset) स्थिर {
        स्थिर uint64_t end_offset = uint64_t(तथ्य_offset) + nbytes_remain;
        अगर (absolute_offset > end_offset) {
            वापस false;
        }

        तथ्य_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        वापस true;
    }

निजी:
    माप_प्रकार read_raw(व्योम * dst, माप_प्रकार size) स्थिर {
        अगर (callback == nullptr || size == 0) {
            वापस 0;
        }

        uint8_t * तथ्य = static_cast<uint8_t *>(dst);
        माप_प्रकार योग_nread = 0;
        bool reached_eof = false;

        जबतक (योग_nread < size) {
            स्थिर माप_प्रकार chunk_आकार = std::min(max_chunk_read, size - योग_nread);
            अगर (तथ्य_offset + योग_nread < तथ्य_offset) {
                अवरोध;
            }
            स्थिर माप_प्रकार nread = callback(userdata, static_cast<व्योम *>(तथ्य + योग_nread), तथ्य_offset + योग_nread, chunk_आकार);
            योग_nread += nread;
            अगर (nread != chunk_आकार) {
                reached_eof = true;
                अवरोध;
            }
        }

        तथ्य_offset += योग_nread;
        GGML_ASSERT(योग_nread <= nbytes_remain);
        nbytes_remain -= योग_nread;

        अगर (reached_eof) {
            nbytes_remain = 0;
        }

        वापस योग_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    व्योम * userdata = nullptr;
    माप_प्रकार max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t तथ्य_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

काष्ठा gguf_context * gguf_init_empty(व्योम) {
    वापस नया gguf_context;
}

ढाँचा<typename T>
bool gguf_read_emplace_helper(स्थिर काष्ठा gguf_reader & gr, std::vector<काष्ठा gguf_kv> & kv, स्थिर std::string & कुंजी, स्थिर bool is_array, स्थिर माप_प्रकार n) {
    अगर (is_array) {
        std::vector<T> value;
        प्रयास {
            अगर (!gr.पढ़ो(value, n)) {
                वापस false;
            }
        } पकड़ो (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, कुंजी.c_str());
            वापस false;
        } पकड़ो (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, कुंजी.c_str());
            वापस false;
        }
        kv.emplace_back(कुंजी, value);
    } अन्यथा {
        T value;
        अगर (!gr.पढ़ो(value)) {
            वापस false;
        }
        kv.emplace_back(कुंजी, value);
    }
    वापस true;
}

अटल काष्ठा gguf_context * gguf_init_from_reader(स्थिर काष्ठा gguf_reader & gr, काष्ठा gguf_init_params params) {
    काष्ठा gguf_context * ctx = नया gguf_context;

    bool ok = true;

    // file magic
    {
        std::vector<अक्षर> magic;
        ok = ok && gr.पढ़ो(magic, 4);

        अगर (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            वापस nullptr;
        }

        क्रम (uint32_t i = 0; i < magic.size(); i++) {
            अगर (magic[i] != GGUF_MAGIC[i]) {
                अक्षर c0 = है_छाप(magic[0]) ? magic[0] : '?';
                अक्षर c1 = है_छाप(magic[1]) ? magic[1] : '?';
                अक्षर c2 = है_छाप(magic[2]) ? magic[2] : '?';
                अक्षर c3 = है_छाप(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                वापस nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    अगर (ok && gr.पढ़ो(ctx->version)) {
        अगर (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = false;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        अगर (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = false;
        }

        अगर (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        अगर (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    } अन्यथा {
        ok = false;
    }

    अगर (ok && gr.पढ़ो(n_tensors)) {
        static_assert(माप(माप_प्रकार) <= 8 && माप(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        अगर (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/माप(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/माप(gguf_tensor_info));
            ok = false;
        }
    } अन्यथा {
        ok = false;
    }

    अगर (ok && gr.पढ़ो(n_kv)) {
        static_assert(माप(माप_प्रकार) <= 8 && माप(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        अगर (n_kv < 0 || n_kv > int64_t(SIZE_MAX/माप(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/माप(gguf_kv));
            ok = false;
        }
    } अन्यथा {
        ok = false;
    }

    अगर (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        वापस nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_कुंजियाँ;

        क्रम (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string कुंजी;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            प्रयास {
                ok = ok && gr.पढ़ो(कुंजी);
            } पकड़ो (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } पकड़ो (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            अगर (ok && कुंजी.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = false;
            }
            अगर (ok && !seen_कुंजियाँ.insert(कुंजी).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, कुंजी.c_str(), i);
                ok = false;
            }
            अगर (!ok) {
                अवरोध;
            }

            ok = ok && gr.पढ़ो(type);
            अगर (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.पढ़ो(type);
                ok = ok && gr.पढ़ो(n);
            }
            अगर (!ok) {
                अवरोध;
            }

            चयन (type) {
                हाल GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<भग्न>      (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<bool>       (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<द्विगुन>     (gr, ctx->kv, कुंजी, is_array, n); अवरोध;
                हाल GGUF_TYPE_ARRAY:
                शेष:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, कुंजी.c_str(), type);
                        ok = false;
                    } अवरोध;
            }
        }

        अगर (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            वापस nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        स्थिर पूर्णांक alignment_idx = gguf_find_कुंजी(ctx, GGUF_कुंजी_GENERAL_ALIGNMENT);
        अगर (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_कुंजी_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            वापस nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        अगर (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            वापस nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    क्रम (int64_t i = 0; ok && i < n_tensors; ++i) {
        काष्ठा gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            प्रयास {
                ok = ok && gr.पढ़ो(name);
            } पकड़ो (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            } पकड़ो (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            अगर (name.लंबाई() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.लंबाई(), GGML_MAX_NAME);
                ok = false;
                अवरोध;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            अगर (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = false;
                अवरोध;
            }
        }
        अगर (!ok) {
            अवरोध;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.पढ़ो(n_dims);
            अगर (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                अवरोध;
            }
            क्रम (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                अगर (j < n_dims) {
                    ok = ok && gr.पढ़ो(info.t.ne[j]);
                }

                // check that all ne are non-negative
                अगर (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    अवरोध;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            अगर (ok) {
                int64_t current_nelements = info.t.ne[0];
                क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; ++i) {
                    अगर (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        अगर (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = false;
                             अवरोध;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 अगर (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     अवरोध;
                 }
            }
        }
        अगर (!ok) {
            अवरोध;
        }

        // tensor type
        {
            ok = ok && gr.पढ़ो(info.t.type);

            // check that tensor type is within defined range
            अगर (info.t.type < 0 || info.t.type >= GGML_TYPE_गिनती) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_गिनती);
                ok = false;
                अवरोध;
            }
            स्थिर माप_प्रकार  type_आकार = ggml_type_आकार(info.t.type);
            स्थिर int64_t blck_आकार = ggml_blck_आकार(info.t.type);

            // check that row size is divisible by block size
            अगर (blck_आकार == 0 || info.t.ne[0] % blck_आकार != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (पूर्णांक) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_आकार);
                ok = false;
                अवरोध;
            }

            // check that the size of the tensor in bytes is representable
            अगर (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_आकार(info.t.type)) > SIZE_MAX/ggml_type_आकार(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = false;
                अवरोध;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_आकार;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_आकार);
            क्रम (पूर्णांक j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        अगर (!ok) {
            अवरोध;
        }

        // tensor data offset within buffer
        ok = ok && gr.पढ़ो(info.offset);

        ctx->info.push_back(info);
    }

    अगर (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        वापस nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    अगर (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        वापस nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        क्रम (माप_प्रकार i = 0; i < ctx->info.size(); ++i) {
            स्थिर gguf_tensor_info & ti = ctx->info[i];
            अगर (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                वापस nullptr;
            }
            स्थिर माप_प्रकार nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            अगर (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                वापस nullptr;
            }
            स्थिर माप_प्रकार padded_आकार = GGML_PAD(nbytes, ctx->alignment);
            अगर (SIZE_MAX - ctx->size < padded_आकार) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_आकार);
                gguf_free(ctx);
                वापस nullptr;
            }
            ctx->size += padded_आकार;
        }
    }

    // load the tensor data only if requested
    अगर (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        माप_प्रकार mem_आकार = 0;
        अगर (params.no_alloc) {
            अगर (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                वापस nullptr;
            }

            स्थिर माप_प्रकार overhead = n_tensors * ggml_tensor_overhead();

            mem_आकार = overhead;
        } अन्यथा {
            अगर ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                वापस nullptr;
            }

            स्थिर माप_प्रकार overhead = (n_tensors + 1) * ggml_tensor_overhead();

            अगर (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                वापस nullptr;
            }

            mem_आकार = overhead + ctx->size;
        }

        काष्ठा ggml_init_params pdata = {
            /*mem_size   =*/ mem_आकार,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        अगर (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            वापस nullptr;
        }

        काष्ठा ggml_context * ctx_तथ्य = *params.ctx;

        काष्ठा ggml_tensor * तथ्य = nullptr;

        अगर (!params.no_alloc) {
            तथ्य = ggml_new_tensor_1d(ctx_तथ्य, GGML_TYPE_I8, ctx->size);

            ok = ok && तथ्य != nullptr;

            अगर (ok) {
                ggml_set_name(तथ्य, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.पढ़ो(तथ्य->तथ्य, ctx->size);

            अगर (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_तथ्य);
                *params.ctx = nullptr;
                gguf_free(ctx);
                वापस nullptr;
            }

            ctx->तथ्य = तथ्य->तथ्य;
        }

        ggml_set_no_alloc(ctx_तथ्य, true);

        // create the tensors
        क्रम (माप_प्रकार i = 0; i < ctx->info.size(); ++i) {
            स्थिर काष्ठा gguf_tensor_info & info = ctx->info[i];

            काष्ठा ggml_tensor * cur = ggml_new_tensor(ctx_तथ्य, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            अगर (!ok) {
                अवरोध;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            अगर (!params.no_alloc) {
                cur->तथ्य = (अक्षर *) तथ्य->तथ्य + info.offset;
            }
        }

        अगर (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_तथ्य);
            *params.ctx = nullptr;
            gguf_free(ctx);
            वापस nullptr;
        }

        ggml_set_no_alloc(ctx_तथ्य, params.no_alloc);
    }

    वापस ctx;
}

काष्ठा gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, व्योम * userdata, माप_प्रकार max_chunk_read, uint64_t max_expected_आकार, काष्ठा gguf_init_params params) {
    अगर (callback == nullptr) {
        वापस nullptr;
    }

    स्थिर काष्ठा gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_आकार);
    वापस gguf_init_from_reader(gr, params);
}

काष्ठा gguf_file_reader {
    खाता * file;
    uint64_t offset;
};

अटल माप_प्रकार gguf_file_reader_callback(व्योम * userdata, व्योम * output, uint64_t offset, माप_प्रकार len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    अगर (reader.offset != offset) {
        अगर (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), शुरू_से) != 0) {
            वापस 0;
        }

        reader.offset = offset;
    }

    स्थिर माप_प्रकार nread = ख_पढ़ो(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    वापस nread;
}

काष्ठा gguf_context * gguf_init_from_file_ptr(खाता * file, काष्ठा gguf_init_params params) {
    अगर (!file) {
        वापस nullptr;
    }

    स्थिर int64_t cur = gguf_ftell(file);
    अगर (cur < 0) {
        वापस nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    स्थिर काष्ठा gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    वापस gguf_init_from_reader(gr, params);
}

काष्ठा gguf_बफर_reader {
    स्थिर uint8_t * तथ्य;
    माप_प्रकार          size;
};

अटल माप_प्रकार gguf_बफर_reader_callback(व्योम * userdata, व्योम * output, uint64_t offset, माप_प्रकार len) {
    GGML_ASSERT(len > 0);

    स्थिर gguf_बफर_reader & reader = *static_cast<gguf_बफर_reader *>(userdata);

    अगर (offset > reader.size || len > reader.size - offset) {
        वापस 0;
    }

    स्थिर माप_प्रकार तथ्य_offset = static_cast<माप_प्रकार>(offset);
    स्थिर माप_प्रकार nread = std::min(len, reader.size - तथ्य_offset);
    स_नकल(static_cast<uint8_t *>(output), reader.तथ्य + तथ्य_offset, nread);
    वापस nread;
}

काष्ठा gguf_context * gguf_init_from_बफर(स्थिर व्योम * तथ्य, माप_प्रकार size, काष्ठा gguf_init_params params) {
    अगर (तथ्य == nullptr || size == 0) {
        वापस nullptr;
    }

    gguf_बफर_reader reader = {
        /*.data = */ static_cast<स्थिर uint8_t *>(तथ्य),
        /*.size = */ size,
    };
    स्थिर काष्ठा gguf_reader gr(gguf_बफर_reader_callback, &reader, SIZE_MAX, 0, size);
    वापस gguf_init_from_reader(gr, params);
}

काष्ठा gguf_context * gguf_init_from_file(स्थिर अक्षर * fname, काष्ठा gguf_init_params params) {
    खाता * file = ggml_fopen(fname, "rb");

    अगर (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, म_त्रुटि(त्रुटि_सं));
        वापस nullptr;
    }

    काष्ठा gguf_context * परिणाम = gguf_init_from_file_ptr(file, params);
    ख_बंद(file);
    वापस परिणाम;
}

व्योम gguf_free(काष्ठा gguf_context * ctx) {
    अगर (ctx == nullptr) {
        वापस;
    }
    मिटाओ ctx;
}

स्थिर अक्षर * gguf_type_name(क्रमागत gguf_type type) {
    स्वतः it = GGUF_TYPE_NAME.find(type);
    वापस it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(स्थिर काष्ठा gguf_context * ctx) {
    वापस ctx->version;
}

माप_प्रकार gguf_get_alignment(स्थिर काष्ठा gguf_context * ctx) {
    वापस ctx->alignment;
}

माप_प्रकार gguf_get_तथ्य_offset(स्थिर काष्ठा gguf_context * ctx) {
    वापस ctx->offset;
}

int64_t gguf_get_n_kv(स्थिर काष्ठा gguf_context * ctx) {
    वापस ctx->kv.size();
}

int64_t gguf_find_कुंजी(स्थिर काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी) {
    // return -1 if key not found
    int64_t keyfound = -1;

    स्थिर int64_t n_kv = gguf_get_n_kv(ctx);

    क्रम (int64_t i = 0; i < n_kv; ++i) {
        अगर (म_भेद(कुंजी, gguf_get_कुंजी(ctx, i)) == 0) {
            keyfound = i;
            अवरोध;
        }
    }

    वापस keyfound;
}

स्थिर अक्षर * gguf_get_कुंजी(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    वापस ctx->kv[कुंजी_id].get_कुंजी().c_str();
}

क्रमागत gguf_type gguf_get_kv_type(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    वापस ctx->kv[कुंजी_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[कुंजी_id].get_type();
}

क्रमागत gguf_type gguf_get_arr_type(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].is_array);
    वापस ctx->kv[कुंजी_id].get_type();
}

स्थिर व्योम * gguf_get_arr_तथ्य(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_type() != GGUF_TYPE_STRING);
    वापस ctx->kv[कुंजी_id].तथ्य.तथ्य();
}

स्थिर अक्षर * gguf_get_arr_str(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id, माप_प्रकार i) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_type() == GGUF_TYPE_STRING);
    वापस ctx->kv[कुंजी_id].तथ्य_string[i].c_str();
}

माप_प्रकार gguf_get_arr_n(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));

    अगर (ctx->kv[कुंजी_id].type == GGUF_TYPE_STRING) {
        वापस ctx->kv[कुंजी_id].तथ्य_string.size();
    }

    स्थिर माप_प्रकार type_आकार = gguf_type_आकार(ctx->kv[कुंजी_id].type);
    GGML_ASSERT(ctx->kv[कुंजी_id].तथ्य.size() % type_आकार == 0);
    वापस ctx->kv[कुंजी_id].तथ्य.size() / type_आकार;
}

uint8_t gguf_get_val_u8(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<int32_t>();
}

भग्न gguf_get_val_f32(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<भग्न>();
}

uint64_t gguf_get_val_u64(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<int64_t>();
}

द्विगुन gguf_get_val_f64(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<द्विगुन>();
}

bool gguf_get_val_bool(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<bool>();
}

स्थिर अक्षर * gguf_get_val_str(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    वापस ctx->kv[कुंजी_id].get_val<std::string>().c_str();
}

स्थिर व्योम * gguf_get_val_तथ्य(स्थिर काष्ठा gguf_context * ctx, int64_t कुंजी_id) {
    GGML_ASSERT(कुंजी_id >= 0 && कुंजी_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुंजी_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[कुंजी_id].get_type() != GGUF_TYPE_STRING);
    वापस ctx->kv[कुंजी_id].तथ्य.तथ्य();
}

int64_t gguf_get_n_tensors(स्थिर काष्ठा gguf_context * ctx) {
    वापस ctx->info.size();
}

int64_t gguf_find_tensor(स्थिर काष्ठा gguf_context * ctx, स्थिर अक्षर * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    स्थिर int64_t n_tensors = gguf_get_n_tensors(ctx);

    क्रम (int64_t i = 0; i < n_tensors; ++i) {
        अगर (म_भेद(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            अवरोध;
        }
    }

    वापस tensor_id;
}

माप_प्रकार gguf_get_tensor_offset(स्थिर काष्ठा gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    वापस ctx->info[tensor_id].offset;
}

स्थिर अक्षर * gguf_get_tensor_name(स्थिर काष्ठा gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    वापस ctx->info[tensor_id].t.name;
}

स्थिर int64_t * gguf_get_tensor_ne(स्थिर काष्ठा gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    वापस ctx->info[tensor_id].t.ne;
}

क्रमागत ggml_type gguf_get_tensor_type(स्थिर काष्ठा gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    वापस ctx->info[tensor_id].t.type;
}

माप_प्रकार gguf_get_tensor_आकार(स्थिर काष्ठा gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    वापस ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_कुंजी(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी) {
    स्थिर int64_t कुंजी_id = gguf_find_कुंजी(ctx, कुंजी);
    अगर (कुंजी_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + कुंजी_id);
    }
    वापस कुंजी_id;
}

ढाँचा<typename T>
अटल व्योम gguf_check_reserved_कुंजियाँ(स्थिर std::string & कुंजी, स्थिर T val) {
    अगर (कुंजी == GGUF_कुंजी_GENERAL_ALIGNMENT) {
        अगर constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_कुंजी_GENERAL_ALIGNMENT " must be power of 2");
        } अन्यथा {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_कुंजी_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

व्योम gguf_set_val_u8(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, uint8_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_i8(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, int8_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_u16(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, uint16_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_i16(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, int16_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_u32(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, uint32_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_i32(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, int32_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_f32(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, भग्न val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_u64(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, uint64_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_i64(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, int64_t val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_f64(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, द्विगुन val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_bool(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, bool val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, val);
}

व्योम gguf_set_val_str(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, स्थिर अक्षर * val) {
    gguf_check_reserved_कुंजियाँ(कुंजी, val);
    gguf_remove_कुंजी(ctx, कुंजी);
    ctx->kv.emplace_back(कुंजी, std::string(val));
}

व्योम gguf_set_arr_तथ्य(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, क्रमागत gguf_type type, स्थिर व्योम * तथ्य, माप_प्रकार n) {
    gguf_check_reserved_कुंजियाँ(कुंजी, तथ्य);
    gguf_remove_कुंजी(ctx, कुंजी);

    स्थिर माप_प्रकार nbytes = n*gguf_type_आकार(type);
    std::vector<int8_t> tmp(nbytes);
    अगर (!tmp.empty()) {
        स_नकल(tmp.तथ्य(), तथ्य, nbytes);
    }
    ctx->kv.emplace_back(कुंजी, tmp);
    ctx->kv.back().cast(type);
}

व्योम gguf_set_arr_str(काष्ठा gguf_context * ctx, स्थिर अक्षर * कुंजी, स्थिर अक्षर ** तथ्य, माप_प्रकार n) {
    gguf_check_reserved_कुंजियाँ(कुंजी, तथ्य);
    gguf_remove_कुंजी(ctx, कुंजी);

    std::vector<std::string> tmp(n);
    क्रम (माप_प्रकार i = 0; i < n; ++i) {
        tmp[i] = तथ्य[i];
    }
    ctx->kv.emplace_back(कुंजी, tmp);
}

// set or add KV pairs from another context
व्योम gguf_set_kv(काष्ठा gguf_context * ctx, स्थिर काष्ठा gguf_context * src) {
    स्थिर int64_t n_kv = gguf_get_n_kv(src);
    क्रम (int64_t i = 0; i < n_kv; ++i) {
        स्थिर काष्ठा gguf_kv & kv = src->kv[i];

        अगर (!kv.is_array) {
            चयन (kv.get_type()) {
                हाल GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_कुंजी().c_str(), kv.get_val<uint8_t>());             अवरोध;
                हाल GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_कुंजी().c_str(), kv.get_val<int8_t>());              अवरोध;
                हाल GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_कुंजी().c_str(), kv.get_val<uint16_t>());            अवरोध;
                हाल GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_कुंजी().c_str(), kv.get_val<int16_t>());             अवरोध;
                हाल GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_कुंजी().c_str(), kv.get_val<uint32_t>());            अवरोध;
                हाल GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_कुंजी().c_str(), kv.get_val<int32_t>());             अवरोध;
                हाल GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_कुंजी().c_str(), kv.get_val<भग्न>());               अवरोध;
                हाल GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_कुंजी().c_str(), kv.get_val<uint64_t>());            अवरोध;
                हाल GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_कुंजी().c_str(), kv.get_val<int64_t>());             अवरोध;
                हाल GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_कुंजी().c_str(), kv.get_val<द्विगुन>());              अवरोध;
                हाल GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_कुंजी().c_str(), kv.get_val<bool>());                अवरोध;
                हाल GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_कुंजी().c_str(), kv.get_val<std::string>().c_str()); अवरोध;
                हाल GGUF_TYPE_ARRAY:
                शेष: GGML_ABORT("invalid type");
            }
            जारी;
        }

        स्थिर माप_प्रकार ne = kv.get_ne();

        चयन (kv.get_type()) {
            हाल GGUF_TYPE_UINT8:
            हाल GGUF_TYPE_INT8:
            हाल GGUF_TYPE_UINT16:
            हाल GGUF_TYPE_INT16:
            हाल GGUF_TYPE_UINT32:
            हाल GGUF_TYPE_INT32:
            हाल GGUF_TYPE_FLOAT32:
            हाल GGUF_TYPE_UINT64:
            हाल GGUF_TYPE_INT64:
            हाल GGUF_TYPE_FLOAT64:
            हाल GGUF_TYPE_BOOL: {
                gguf_set_arr_तथ्य(ctx, kv.get_कुंजी().c_str(), kv.get_type(), kv.तथ्य.तथ्य(), ne);
            } अवरोध;
            हाल GGUF_TYPE_STRING: {
                std::vector<स्थिर अक्षर *> tmp(ne);
                क्रम (माप_प्रकार j = 0; j < ne; ++j) {
                    tmp[j] = kv.तथ्य_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_कुंजी().c_str(), tmp.तथ्य(), ne);
            } अवरोध;
            हाल GGUF_TYPE_ARRAY:
            शेष: GGML_ABORT("invalid type");
        }
    }
}

व्योम gguf_add_tensor(
             काष्ठा gguf_context * ctx,
        स्थिर काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    अगर (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    काष्ठा gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

व्योम gguf_set_tensor_type(काष्ठा gguf_context * ctx, स्थिर अक्षर * name, क्रमागत ggml_type type) {
    स्थिर int64_t tensor_id = gguf_find_tensor(ctx, name);
    अगर (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    काष्ठा ggml_tensor * tensor = &ctx->info[tensor_id].t;
    स्थिर माप_प्रकार  type_आकार = ggml_type_आकार(type);
    स्थिर int64_t blck_आकार = ggml_blck_आकार(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_आकार == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_आकार;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_आकार);
    क्रम (पूर्णांक i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    स्थिर int64_t n_tensors = gguf_get_n_tensors(ctx);
    क्रम (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

व्योम gguf_set_tensor_तथ्य(काष्ठा gguf_context * ctx, स्थिर अक्षर * name, स्थिर व्योम * तथ्य) {
    स्थिर int64_t tensor_id = gguf_find_tensor(ctx, name);
    अगर (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.तथ्य = (व्योम *)(uintptr_t)तथ्य; // double cast suppresses warning about casting away const
}

काष्ठा gguf_writer_base {
    माप_प्रकार written_bytes {0u};

    भव ~gguf_writer_base(व्योम) = शेष;

    // we bet on devirtualization
    भव व्योम ग_लिखो(int8_t val) = 0;
    भव व्योम ग_लिखो(स्थिर std::vector<int8_t> & val) = 0;
    भव व्योम write_tensor_तथ्य(स्थिर काष्ठा gguf_tensor_info & info, माप_प्रकार offset_तथ्य, माप_प्रकार alignment) = 0;

    ढाँचा <typename T>
    व्योम ग_लिखो(स्थिर T & val) {
        क्रम (माप_प्रकार i = 0; i < माप(val); ++i) {
            ग_लिखो(reinterpret_cast<स्थिर int8_t *>(&val)[i]);
        }
    }

    व्योम ग_लिखो(स्थिर bool & val) {
        स्थिर int8_t val8 = val ? 1 : 0;
        ग_लिखो(val8);
    }

    व्योम ग_लिखो(स्थिर std::string & val) {
        {
            स्थिर uint64_t n = val.लंबाई();
            ग_लिखो(n);
        }
        क्रम (माप_प्रकार i = 0; i < val.लंबाई(); ++i) {
            ग_लिखो((val.तथ्य())[i]);
        }
    }

    व्योम ग_लिखो(स्थिर अक्षर * val) {
        ग_लिखो(std::string(val));
    }

    व्योम ग_लिखो(स्थिर क्रमागत ggml_type & val) {
        ग_लिखो(int32_t(val));
    }

    व्योम ग_लिखो(स्थिर क्रमागत gguf_type & val) {
        ग_लिखो(int32_t(val));
    }

    व्योम ग_लिखो(स्थिर काष्ठा gguf_kv & kv) {
        स्थिर uint64_t ne = kv.get_ne();

        ग_लिखो(kv.get_कुंजी());

        अगर (kv.is_array) {
            ग_लिखो(GGUF_TYPE_ARRAY);
            ग_लिखो(kv.get_type());
            ग_लिखो(ne);
        } अन्यथा {
            ग_लिखो(kv.get_type());
        }

        चयन (kv.get_type()) {
            हाल GGUF_TYPE_UINT8:
            हाल GGUF_TYPE_INT8:
            हाल GGUF_TYPE_UINT16:
            हाल GGUF_TYPE_INT16:
            हाल GGUF_TYPE_UINT32:
            हाल GGUF_TYPE_INT32:
            हाल GGUF_TYPE_FLOAT32:
            हाल GGUF_TYPE_UINT64:
            हाल GGUF_TYPE_INT64:
            हाल GGUF_TYPE_FLOAT64: {
                ग_लिखो(kv.तथ्य);
            } अवरोध;
            हाल GGUF_TYPE_BOOL: {
                क्रम (माप_प्रकार i = 0; i < ne; ++i) {
                    ग_लिखो(kv.get_val<bool>(i));
                }
            } अवरोध;
            हाल GGUF_TYPE_STRING: {
                क्रम (माप_प्रकार i = 0; i < ne; ++i) {
                    ग_लिखो(kv.get_val<std::string>(i));
                }
            } अवरोध;
            हाल GGUF_TYPE_ARRAY:
            शेष: GGML_ABORT("invalid type");
        }
    }

    व्योम write_tensor_meta(स्थिर काष्ठा gguf_tensor_info & info) {
        ग_लिखो(info.t.name);

        स्थिर uint32_t n_dims = ggml_n_dims(&info.t);
        ग_लिखो(n_dims);

        क्रम (uint32_t j = 0; j < n_dims; ++j) {
            ग_लिखो(info.t.ne[j]);
        }
        ग_लिखो(info.t.type);
        ग_लिखो(info.offset);
    }

    व्योम pad(स्थिर माप_प्रकार alignment) {
        जबतक (written_bytes % alignment != 0) {
            स्थिर int8_t zero = 0;
            ग_लिखो(zero);
        }
    }
};

// vector buffer based writer
काष्ठा gguf_writer_buf final : खुला gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::ग_लिखो;

    व्योम ग_लिखो(स्थिर int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    व्योम ग_लिखो(स्थिर std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    व्योम write_tensor_तथ्य(स्थिर काष्ठा gguf_tensor_info & info, स्थिर माप_प्रकार offset_तथ्य, स्थिर माप_प्रकार alignment) override {
        GGML_ASSERT(buf.size() - offset_तथ्य == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        स्थिर माप_प्रकार offset = buf.size();
        स्थिर माप_प्रकार nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        अगर (info.t.बफर) {
            ggml_backend_tensor_get(&info.t, buf.तथ्य() + offset, 0, nbytes);
        } अन्यथा {
            GGML_ASSERT(info.t.तथ्य);
            स_नकल(buf.तथ्य() + offset, info.t.तथ्य, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
काष्ठा gguf_writer_file final : खुला gguf_writer_base {
    खाता * file;

    gguf_writer_file(खाता* file) : file(file) {}

    using gguf_writer_base::ग_लिखो;

    व्योम ग_लिखो(स्थिर int8_t val) override {
        स्थिर स्वतः real_val = static_cast<uint8_t>(val);
        स्थिर स्वतः ret = ख_अक्षर_दो(real_val, file);
        written_bytes++;
        अगर (ret != real_val) {
            फेंको std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((पूर्णांक)real_val) + "'");
        }
    }

    व्योम ग_लिखो(स्थिर std::vector<int8_t> & val) override {
        स्थिर स्वतः ret = ख_डालो(val.तथ्य(), 1, val.size(), file);
        written_bytes += val.size();
        अगर (ret != val.size()) {
            फेंको std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    व्योम write_tensor_तथ्य(स्थिर काष्ठा gguf_tensor_info & info, स्थिर माप_प्रकार offset_तथ्य, स्थिर माप_प्रकार alignment) override {
        GGML_ASSERT(written_bytes - offset_तथ्य == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        स्थिर माप_प्रकार nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        अगर (info.t.बफर) {
            ggml_backend_tensor_get(&info.t, buf.तथ्य(), 0, nbytes);
        } अन्यथा {
            GGML_ASSERT(info.t.तथ्य);
            स_नकल(buf.तथ्य(), info.t.तथ्य, nbytes);
        }
        ग_लिखो(buf);

        pad(alignment);
    }
};

ढाँचा <typename writer_t>
अटल व्योम gguf_write_out(स्थिर काष्ठा gguf_context * ctx, writer_t & gw, bool only_meta) {
    स्थिर int64_t n_kv      = gguf_get_n_kv(ctx);
    स्थिर int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.ग_लिखो(GGUF_MAGIC[0]);
    gw.ग_लिखो(GGUF_MAGIC[1]);
    gw.ग_लिखो(GGUF_MAGIC[2]);
    gw.ग_लिखो(GGUF_MAGIC[3]);
    gw.ग_लिखो(ctx->version);
    gw.ग_लिखो(n_tensors);
    gw.ग_लिखो(n_kv);

    // write key-value pairs
    क्रम (int64_t i = 0; i < n_kv; ++i) {
        gw.ग_लिखो(ctx->kv[i]);
    }

    // write tensor info
    क्रम (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    अगर (only_meta) {
        वापस;
    }

    स्थिर माप_प्रकार offset_तथ्य = gw.written_bytes;

    // write tensor data
    क्रम (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_तथ्य(ctx->info[i], offset_तथ्य, ctx->alignment);
    }
}

व्योम gguf_write_to_buf(स्थिर काष्ठा gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

bool gguf_write_to_file_ptr(स्थिर काष्ठा gguf_context * ctx, खाता * file, bool only_meta) {
    GGML_ASSERT(file);

    प्रयास {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } पकड़ो (स्थिर std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        वापस false;
    }
    वापस true;
}

bool gguf_write_to_file(स्थिर काष्ठा gguf_context * ctx, स्थिर अक्षर * fname, bool only_meta) {
    खाता * file = ggml_fopen(fname, "wb");

    अगर (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        वापस false;
    }

    स्थिर bool success = gguf_write_to_file_ptr(ctx, file, only_meta);
    अगर (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    ख_बंद(file);
    वापस success;
}

माप_प्रकार gguf_get_meta_आकार(स्थिर काष्ठा gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    वापस buf.size();
}

व्योम gguf_get_meta_तथ्य(स्थिर काष्ठा gguf_context * ctx, व्योम * तथ्य) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    स_नकल(तथ्य, buf.तथ्य(), buf.size());
}
