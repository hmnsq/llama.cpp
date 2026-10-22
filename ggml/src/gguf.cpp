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

#परिभाषय GGUF_MAX_STRING_दीर्घता  (1024*1024*1024)
#परिभाषय GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#यदि_परिभाषितम् _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#अन्यथा
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#यदि_अन्तः

प्रतिरूपम् <प्रकारनाम T>
संरचना type_गन्तव्यम्_gguf_type;

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<uint8_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT8;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<int8_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT8;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<uint16_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT16;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<int16_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT16;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<uint32_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT32;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<int32_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT32;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<भिन्न> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_FLOAT32;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<तार्किक> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_BOOL;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<std::string> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_STRING;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<uint64_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_UINT64;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<int64_t> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_INT64;
};

प्रतिरूपम् <>
संरचना type_गन्तव्यम्_gguf_type<द्विगुण> {
    स्थिर constexpr गणना gguf_type value = GGUF_TYPE_FLOAT64;
};

स्थिर ध्रुव std::map<gguf_type, size_t> GGUF_TYPE_आकार = {
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
static_assert(GGUF_TYPE_गणनम् == 13, "GGUF_TYPE_COUNT != 13");

स्थिर ध्रुव std::map<gguf_type, ध्रुव वर्ण *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_गणनम् == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_आकार(गणना gguf_type type) {
    स्वचालित it = GGUF_TYPE_आकार.find(type);
    प्रत्यागम it == GGUF_TYPE_आकार.end() ? 0 : it->second;
}

संरचना gguf_kv {
    std::string कुञ्चिका;

    तार्किक is_array;
    गणना gguf_type type;

    std::vector<int8_t>      दत्त;
    std::vector<std::string> दत्त_string;

    प्रतिरूपम् <प्रकारनाम T>
    gguf_kv(ध्रुव std::string & कुञ्चिका, ध्रुव T value)
            : कुञ्चिका(कुञ्चिका), is_array(असत्य), type(type_गन्तव्यम्_gguf_type<T>::value) {
        GGML_ASSERT(!कुञ्चिका.रिक्तानि());
        दत्त.resize(परिमाण(T));
        स्मृतिं_प्रतिलिख(दत्त.दत्त(), &value, परिमाण(T));
    }

    प्रतिरूपम् <प्रकारनाम T>
    gguf_kv(ध्रुव std::string & कुञ्चिका, ध्रुव std::vector<T> & value)
            : कुञ्चिका(कुञ्चिका), is_array(सत्य), type(type_गन्तव्यम्_gguf_type<T>::value) {
        GGML_ASSERT(!कुञ्चिका.रिक्तानि());
        दत्त.resize(value.size()*परिमाण(T));
        प्रति (size_t i = 0; i < value.size(); ++i) {
            ध्रुव T tmp = value[i];
            स्मृतिं_प्रतिलिख(दत्त.दत्त() + i*परिमाण(T), &tmp, परिमाण(T));
        }
    }

    gguf_kv(ध्रुव std::string & कुञ्चिका, ध्रुव std::string & value)
            : कुञ्चिका(कुञ्चिका), is_array(असत्य), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!कुञ्चिका.रिक्तानि());
        दत्त_string.push_back(value);
    }

    gguf_kv(ध्रुव std::string & कुञ्चिका, ध्रुव std::vector<std::string> & value)
            : कुञ्चिका(कुञ्चिका), is_array(सत्य), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!कुञ्चिका.रिक्तानि());
        दत्त_string = value;
    }

    ध्रुव std::string & get_कुञ्चिका() ध्रुव {
        प्रत्यागम कुञ्चिका;
    }

    ध्रुव गणना gguf_type & get_type() ध्रुव {
        प्रत्यागम type;
    }

    size_t get_ne() ध्रुव {
        यदि (type == GGUF_TYPE_STRING) {
            ध्रुव size_t ne = दत्त_string.size();
            GGML_ASSERT(is_array || ne == 1);
            प्रत्यागम ne;
        }
        ध्रुव size_t type_आकार = gguf_type_आकार(type);
        GGML_ASSERT(दत्त.size() % type_आकार == 0);
        ध्रुव size_t ne = दत्त.size() / type_आकार;
        GGML_ASSERT(is_array || ne == 1);
        प्रत्यागम ne;
    }

    प्रतिरूपम् <प्रकारनाम T>
    ध्रुव T & get_val(ध्रुव size_t i = 0) ध्रुव {
        GGML_ASSERT(type_गन्तव्यम्_gguf_type<T>::value == type);
        यदि constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(दत्त_string.size() >= i+1);
            प्रत्यागम दत्त_string[i];
        }
        ध्रुव size_t type_आकार = gguf_type_आकार(type);
        GGML_ASSERT(दत्त.size() % type_आकार == 0);
        GGML_ASSERT(दत्त.size() >= (i+1)*type_आकार);
        प्रत्यागम reinterpret_cast<ध्रुव T *>(दत्त.दत्त())[i];
    }

    शून्य cast(ध्रुव गणना gguf_type new_type) {
        ध्रुव size_t new_type_आकार = gguf_type_आकार(new_type);
        GGML_ASSERT(दत्त.size() % new_type_आकार == 0);
        type = new_type;
    }
};

संरचना gguf_tensor_info {
    संरचना ggml_tensor t; // for holding the equivalent info
    uint64_t विस्थापनम्;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

संरचना gguf_context {
    uint32_t संस्करणम्_2 = GGUF_संस्करणम्;

    std::vector<संरचना gguf_kv> kv;
    std::vector<संरचना gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t विस्थापनम्    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    शून्य * दत्त = शून्यसूचकः;
};

संरचना gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            शून्य * userdata,
            size_t max_chunk_read,
            uint64_t दत्त_विस्थापनम् = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          start_विस्थापनम्(दत्त_विस्थापनम्),
          दत्त_विस्थापनम्(दत्त_विस्थापनम्),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    स्थिर uint64_t file_remain(FILE * file) {
        ध्रुव int64_t cur = gguf_ftell(file);
        यदि (cur < 0) {
            प्रत्यागम 0;
        }
        यदि (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            प्रत्यागम 0;
        }
        ध्रुव int64_t end = gguf_ftell(file);
        यदि (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            प्रत्यागम 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        प्रत्यागम static_cast<uint64_t>(end - cur);
    }

    प्रतिरूपम् <प्रकारनाम T>
    तार्किक पठ(T & dst) ध्रुव {
        ध्रुव size_t size = परिमाण(dst);
        यदि (size > nbytes_remain) {
            प्रत्यागम असत्य;
        }
        प्रत्यागम read_raw(&dst, size) == size;
    }

    प्रतिरूपम् <प्रकारनाम T>
    तार्किक पठ(std::vector<T> & dst, ध्रुव size_t n) ध्रुव {
        यदि (n > GGUF_MAX_ARRAY_ELEMENTS) {
            प्रत्यागम असत्य;
        }
        यदि constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            यदि (n > SIZE_MAX / परिमाण(uint64_t)) {
                प्रत्यागम असत्य;
            }
            यदि (nbytes_remain < n * परिमाण(uint64_t)) {
                प्रत्यागम असत्य;
            }
        } अन्यथा {
            यदि (n > SIZE_MAX / परिमाण(T)) {
                प्रत्यागम असत्य;
            }
            यदि (nbytes_remain < n * परिमाण(T)) {
                प्रत्यागम असत्य;
            }
        }
        dst.resize(n);
        प्रति (size_t i = 0; i < dst.size(); ++i) {
            यदि constexpr (std::is_same<T, तार्किक>::value) {
                तार्किक tmp;
                यदि (!पठ(tmp)) {
                    प्रत्यागम असत्य;
                }
                dst[i] = tmp;
            } अन्यथा {
                यदि (!पठ(dst[i])) {
                    प्रत्यागम असत्य;
                }
            }
        }
        प्रत्यागम सत्य;
    }

    तार्किक पठ(तार्किक & dst) ध्रुव {
        int8_t tmp = -1;
        यदि (!पठ(tmp)) {
            प्रत्यागम असत्य;
        }
        dst = tmp != 0;
        प्रत्यागम सत्य;
    }

    तार्किक पठ(गणना ggml_type & dst) ध्रुव {
        int32_t tmp = -1;
        यदि (!पठ(tmp)) {
            प्रत्यागम असत्य;
        }
        dst = ggml_type(tmp);
        प्रत्यागम सत्य;
    }

    तार्किक पठ(गणना gguf_type & dst) ध्रुव {
        int32_t tmp = -1;
        यदि (!पठ(tmp)) {
            प्रत्यागम असत्य;
        }
        dst = gguf_type(tmp);
        प्रत्यागम सत्य;
    }

    तार्किक पठ(std::string & dst) ध्रुव {
        uint64_t size = 0;
        यदि (!पठ(size)) {
            प्रत्यागम असत्य;
        }
        यदि (size > GGUF_MAX_STRING_दीर्घता) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_दीर्घता);
            प्रत्यागम असत्य;
        }
        यदि (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            प्रत्यागम असत्य;
        }
        dst.resize(static_cast<size_t>(size));
        प्रत्यागम read_raw(dst.दत्त(), static_cast<size_t>(size)) == size;
    }

    तार्किक पठ(शून्य * dst, ध्रुव size_t size) ध्रुव {
        यदि (size > nbytes_remain) {
            प्रत्यागम असत्य;
        }
        प्रत्यागम read_raw(dst, size) == size;
    }

    uint64_t tell() ध्रुव {
        प्रत्यागम दत्त_विस्थापनम्;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ध्रुव {
        प्रत्यागम start_विस्थापनम्;
    }

    तार्किक seek(uint64_t absolute_विस्थापनम्) ध्रुव {
        ध्रुव uint64_t अन्तः_विस्थापनम् = uint64_t(दत्त_विस्थापनम्) + nbytes_remain;
        यदि (absolute_विस्थापनम् > अन्तः_विस्थापनम्) {
            प्रत्यागम असत्य;
        }

        दत्त_विस्थापनम् = absolute_विस्थापनम्;
        nbytes_remain = अन्तः_विस्थापनम् - absolute_विस्थापनम्;

        प्रत्यागम सत्य;
    }

गुप्त:
    size_t read_raw(शून्य * dst, size_t size) ध्रुव {
        यदि (callback == शून्यसूचकः || size == 0) {
            प्रत्यागम 0;
        }

        uint8_t * दत्त = static_cast<uint8_t *>(dst);
        size_t योग_nread = 0;
        तार्किक reached_eof = असत्य;

        यावत् (योग_nread < size) {
            ध्रुव size_t chunk_आकार = std::min(max_chunk_read, size - योग_nread);
            यदि (दत्त_विस्थापनम् + योग_nread < दत्त_विस्थापनम्) {
                भङ्ग;
            }
            ध्रुव size_t nread = callback(userdata, static_cast<शून्य *>(दत्त + योग_nread), दत्त_विस्थापनम् + योग_nread, chunk_आकार);
            योग_nread += nread;
            यदि (nread != chunk_आकार) {
                reached_eof = सत्य;
                भङ्ग;
            }
        }

        दत्त_विस्थापनम् += योग_nread;
        GGML_ASSERT(योग_nread <= nbytes_remain);
        nbytes_remain -= योग_nread;

        यदि (reached_eof) {
            nbytes_remain = 0;
        }

        प्रत्यागम योग_nread;
    }

    gguf_reader_callback_t callback = शून्यसूचकः;
    शून्य * userdata = शून्यसूचकः;
    size_t max_chunk_read = 0;
    uint64_t start_विस्थापनम् = 0;
    परिवर्तनीय uint64_t दत्त_विस्थापनम् = 0;
    परिवर्तनीय uint64_t nbytes_remain = 0;
};

संरचना gguf_context * gguf_init_रिक्तानि(शून्य) {
    प्रत्यागम नूतन gguf_context;
}

प्रतिरूपम्<प्रकारनाम T>
तार्किक gguf_read_emplace_helper(ध्रुव संरचना gguf_reader & gr, std::vector<संरचना gguf_kv> & kv, ध्रुव std::string & कुञ्चिका, ध्रुव तार्किक is_array, ध्रुव size_t n) {
    यदि (is_array) {
        std::vector<T> value;
        यत्न {
            यदि (!gr.पठ(value, n)) {
                प्रत्यागम असत्य;
            }
        } गृहाण (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, कुञ्चिका.c_str());
            प्रत्यागम असत्य;
        } गृहाण (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, कुञ्चिका.c_str());
            प्रत्यागम असत्य;
        }
        kv.emplace_back(कुञ्चिका, value);
    } अन्यथा {
        T value;
        यदि (!gr.पठ(value)) {
            प्रत्यागम असत्य;
        }
        kv.emplace_back(कुञ्चिका, value);
    }
    प्रत्यागम सत्य;
}

स्थिर संरचना gguf_context * gguf_init_from_reader(ध्रुव संरचना gguf_reader & gr, संरचना gguf_init_params params) {
    संरचना gguf_context * ctx = नूतन gguf_context;

    तार्किक ok = सत्य;

    // file magic
    {
        std::vector<वर्ण> magic;
        ok = ok && gr.पठ(magic, 4);

        यदि (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            प्रत्यागम शून्यसूचकः;
        }

        प्रति (uint32_t i = 0; i < magic.size(); i++) {
            यदि (magic[i] != GGUF_MAGIC[i]) {
                वर्ण c0 = isprint(magic[0]) ? magic[0] : '?';
                वर्ण c1 = isprint(magic[1]) ? magic[1] : '?';
                वर्ण c2 = isprint(magic[2]) ? magic[2] : '?';
                वर्ण c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }
        }
    }

    //शीर्षकम्
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    यदि (ok && gr.पठ(ctx->संस्करणम्_2)) {
        यदि (ok && ctx->संस्करणम्_2 == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->संस्करणम्_2);
            ok = असत्य;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        यदि (ok && (ctx->संस्करणम्_2 & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->संस्करणम्_2);
            ok = असत्य;
        }

        यदि (ok && ctx->संस्करणम्_2 == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = असत्य;
        }
        यदि (ok && ctx->संस्करणम्_2 > GGUF_संस्करणम्) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->संस्करणम्_2, GGUF_संस्करणम्);
            ok = असत्य;
        }
    } अन्यथा {
        ok = असत्य;
    }

    यदि (ok && gr.पठ(n_tensors)) {
        static_assert(परिमाण(size_t) <= 8 && परिमाण(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        यदि (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/परिमाण(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/परिमाण(gguf_tensor_info));
            ok = असत्य;
        }
    } अन्यथा {
        ok = असत्य;
    }

    यदि (ok && gr.पठ(n_kv)) {
        static_assert(परिमाण(size_t) <= 8 && परिमाण(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
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
        प्रत्यागम शून्यसूचकः;
    }

    // KV pairs
    {
        std::unordered_set<std::string> आलोकितम्_कुञ्चिकाः;

        प्रति (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string कुञ्चिका;
            gguf_type   type     = gguf_type(-1);
            तार्किक        is_array = असत्य;
            uint64_t    n        = 1;

            यत्न {
                ok = ok && gr.पठ(कुञ्चिका);
            } गृहाण (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = असत्य;
            } गृहाण (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = असत्य;
            }
            यदि (ok && कुञ्चिका.रिक्तानि()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = असत्य;
            }
            यदि (ok && !आलोकितम्_कुञ्चिकाः.insert(कुञ्चिका).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, कुञ्चिका.c_str(), i);
                ok = असत्य;
            }
            यदि (!ok) {
                भङ्ग;
            }

            ok = ok && gr.पठ(type);
            यदि (type == GGUF_TYPE_ARRAY) {
                is_array = सत्य;
                ok = ok && gr.पठ(type);
                ok = ok && gr.पठ(n);
            }
            यदि (!ok) {
                भङ्ग;
            }

            विकल्प (type) {
                पक्ष GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<भिन्न>      (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<तार्किक>       (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<द्विगुण>     (gr, ctx->kv, कुञ्चिका, is_array, n); भङ्ग;
                पक्ष GGUF_TYPE_ARRAY:
                स्वभाव:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, कुञ्चिका.c_str(), type);
                        ok = असत्य;
                    } भङ्ग;
            }
        }

        यदि (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            प्रत्यागम शून्यसूचकः;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ध्रुव पूर्णाङ्क alignment_idx = gguf_find_कुञ्चिका(ctx, GGUF_कुञ्चिका_GENERAL_ALIGNMENT);
        यदि (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_कुञ्चिका_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            प्रत्यागम शून्यसूचकः;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        यदि (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            प्रत्यागम शून्यसूचकः;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> आलोकितम्_tensor_नामानि;

    प्रति (int64_t i = 0; ok && i < n_tensors; ++i) {
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
            यदि (name.दीर्घता() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.दीर्घता(), GGML_MAX_NAME);
                ok = असत्य;
                भङ्ग;
            }
            ggml_समुच्चयः_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            यदि (ok && !आलोकितम्_tensor_नामानि.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = असत्य;
                भङ्ग;
            }
        }
        यदि (!ok) {
            भङ्ग;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.पठ(n_dims);
            यदि (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = असत्य;
                भङ्ग;
            }
            प्रति (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                यदि (j < n_dims) {
                    ok = ok && gr.पठ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                यदि (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = असत्य;
                    भङ्ग;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            यदि (ok) {
                int64_t current_nelements = info.t.ne[0];
                प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                    यदि (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        यदि (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = असत्य;
                             भङ्ग;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 यदि (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     भङ्ग;
                 }
            }
        }
        यदि (!ok) {
            भङ्ग;
        }

        // tensor type
        {
            ok = ok && gr.पठ(info.t.type);

            // check that tensor type is within defined range
            यदि (info.t.type < 0 || info.t.type >= GGML_TYPE_गणनम्) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_गणनम्);
                ok = असत्य;
                भङ्ग;
            }
            ध्रुव size_t  type_आकार = ggml_type_आकार(info.t.type);
            ध्रुव int64_t blck_आकार = ggml_blck_आकार(info.t.type);

            // check that row size is divisible by block size
            यदि (blck_आकार == 0 || info.t.ne[0] % blck_आकार != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (पूर्णाङ्क) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_आकार);
                ok = असत्य;
                भङ्ग;
            }

            // check that the size of the tensor in bytes is representable
            यदि (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_आकार(info.t.type)) > SIZE_MAX/ggml_type_आकार(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = असत्य;
                भङ्ग;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_आकार;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_आकार);
            प्रति (पूर्णाङ्क j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        यदि (!ok) {
            भङ्ग;
        }

        // tensor data offset within buffer
        ok = ok && gr.पठ(info.विस्थापनम्);

        ctx->info.push_back(info);
    }

    यदि (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        प्रत्यागम शून्यसूचकः;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    यदि (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        प्रत्यागम शून्यसूचकः;
    }

    // store the current file offset - this is where the data section starts
    ctx->विस्थापनम् = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        प्रति (size_t i = 0; i < ctx->info.size(); ++i) {
            ध्रुव gguf_tensor_info & ti = ctx->info[i];
            यदि (ti.विस्थापनम् != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.विस्थापनम्, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }
            ध्रुव size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            यदि (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }
            ध्रुव size_t padded_आकार = GGML_PAD(nbytes, ctx->alignment);
            यदि (SIZE_MAX - ctx->size < padded_आकार) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_आकार);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }
            ctx->size += padded_आकार;
        }
    }

    // load the tensor data only if requested
    यदि (params.ctx != शून्यसूचकः) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_आकार = 0;
        यदि (params.no_alloc) {
            यदि (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }

            ध्रुव size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_आकार = overhead;
        } अन्यथा {
            यदि ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }

            ध्रुव size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            यदि (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }

            mem_आकार = overhead + ctx->size;
        }

        संरचना ggml_init_params pdata = {
            /*mem_size   =*/ mem_आकार,
            /*mem_buffer =*/ शून्यसूचकः,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        यदि (*params.ctx == शून्यसूचकः) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            प्रत्यागम शून्यसूचकः;
        }

        संरचना ggml_context * ctx_दत्त = *params.ctx;

        संरचना ggml_tensor * दत्त = शून्यसूचकः;

        यदि (!params.no_alloc) {
            दत्त = ggml_new_tensor_1d(ctx_दत्त, GGML_TYPE_I8, ctx->size);

            ok = ok && दत्त != शून्यसूचकः;

            यदि (ok) {
                ggml_समुच्चयः_name(दत्त, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.पठ(दत्त->दत्त, ctx->size);

            यदि (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_दत्त);
                *params.ctx = शून्यसूचकः;
                gguf_free(ctx);
                प्रत्यागम शून्यसूचकः;
            }

            ctx->दत्त = दत्त->दत्त;
        }

        ggml_समुच्चयः_no_alloc(ctx_दत्त, सत्य);

        // create the tensors
        प्रति (size_t i = 0; i < ctx->info.size(); ++i) {
            ध्रुव संरचना gguf_tensor_info & info = ctx->info[i];

            संरचना ggml_tensor * cur = ggml_new_tensor(ctx_दत्त, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != शून्यसूचकः;

            यदि (!ok) {
                भङ्ग;
            }

            ggml_समुच्चयः_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            यदि (!params.no_alloc) {
                cur->दत्त = (वर्ण *) दत्त->दत्त + info.विस्थापनम्;
            }
        }

        यदि (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_दत्त);
            *params.ctx = शून्यसूचकः;
            gguf_free(ctx);
            प्रत्यागम शून्यसूचकः;
        }

        ggml_समुच्चयः_no_alloc(ctx_दत्त, params.no_alloc);
    }

    प्रत्यागम ctx;
}

संरचना gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, शून्य * userdata, size_t max_chunk_read, uint64_t max_expected_आकार, संरचना gguf_init_params params) {
    यदि (callback == शून्यसूचकः) {
        प्रत्यागम शून्यसूचकः;
    }

    ध्रुव संरचना gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_आकार);
    प्रत्यागम gguf_init_from_reader(gr, params);
}

संरचना gguf_file_reader {
    FILE * file;
    uint64_t विस्थापनम्;
};

स्थिर size_t gguf_file_reader_callback(शून्य * userdata, शून्य * output, uint64_t विस्थापनम्, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    यदि (reader.विस्थापनम् != विस्थापनम्) {
        यदि (विस्थापनम् > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(विस्थापनम्), SEEK_SET) != 0) {
            प्रत्यागम 0;
        }

        reader.विस्थापनम् = विस्थापनम्;
    }

    ध्रुव size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.विस्थापनम् += nread;
    प्रत्यागम nread;
}

संरचना gguf_context * gguf_init_from_file_ptr(FILE * file, संरचना gguf_init_params params) {
    यदि (!file) {
        प्रत्यागम शून्यसूचकः;
    }

    ध्रुव int64_t cur = gguf_ftell(file);
    यदि (cur < 0) {
        प्रत्यागम शून्यसूचकः;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ध्रुव संरचना gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.विस्थापनम्, gguf_reader::file_remain(file));
    प्रत्यागम gguf_init_from_reader(gr, params);
}

संरचना gguf_बफर_reader {
    ध्रुव uint8_t * दत्त;
    size_t          size;
};

स्थिर size_t gguf_बफर_reader_callback(शून्य * userdata, शून्य * output, uint64_t विस्थापनम्, size_t len) {
    GGML_ASSERT(len > 0);

    ध्रुव gguf_बफर_reader & reader = *static_cast<gguf_बफर_reader *>(userdata);

    यदि (विस्थापनम् > reader.size || len > reader.size - विस्थापनम्) {
        प्रत्यागम 0;
    }

    ध्रुव size_t दत्त_विस्थापनम् = static_cast<size_t>(विस्थापनम्);
    ध्रुव size_t nread = std::min(len, reader.size - दत्त_विस्थापनम्);
    स्मृतिं_प्रतिलिख(static_cast<uint8_t *>(output), reader.दत्त + दत्त_विस्थापनम्, nread);
    प्रत्यागम nread;
}

संरचना gguf_context * gguf_init_from_बफर(ध्रुव शून्य * दत्त, size_t size, संरचना gguf_init_params params) {
    यदि (दत्त == शून्यसूचकः || size == 0) {
        प्रत्यागम शून्यसूचकः;
    }

    gguf_बफर_reader reader = {
        /*.data = */ static_cast<ध्रुव uint8_t *>(दत्त),
        /*.size = */ size,
    };
    ध्रुव संरचना gguf_reader gr(gguf_बफर_reader_callback, &reader, SIZE_MAX, 0, size);
    प्रत्यागम gguf_init_from_reader(gr, params);
}

संरचना gguf_context * gguf_init_from_file(ध्रुव वर्ण * fname, संरचना gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    यदि (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        प्रत्यागम शून्यसूचकः;
    }

    संरचना gguf_context * परिणाम = gguf_init_from_file_ptr(file, params);
    fclose(file);
    प्रत्यागम परिणाम;
}

शून्य gguf_free(संरचना gguf_context * ctx) {
    यदि (ctx == शून्यसूचकः) {
        प्रत्यागम;
    }
    विलोपय ctx;
}

ध्रुव वर्ण * gguf_type_name(गणना gguf_type type) {
    स्वचालित it = GGUF_TYPE_NAME.find(type);
    प्रत्यागम it == GGUF_TYPE_NAME.end() ? शून्यसूचकः : it->second;
}

uint32_t gguf_get_संस्करणम्(ध्रुव संरचना gguf_context * ctx) {
    प्रत्यागम ctx->संस्करणम्_2;
}

size_t gguf_get_alignment(ध्रुव संरचना gguf_context * ctx) {
    प्रत्यागम ctx->alignment;
}

size_t gguf_get_दत्त_विस्थापनम्(ध्रुव संरचना gguf_context * ctx) {
    प्रत्यागम ctx->विस्थापनम्;
}

int64_t gguf_get_n_kv(ध्रुव संरचना gguf_context * ctx) {
    प्रत्यागम ctx->kv.size();
}

int64_t gguf_find_कुञ्चिका(ध्रुव संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ध्रुव int64_t n_kv = gguf_get_n_kv(ctx);

    प्रति (int64_t i = 0; i < n_kv; ++i) {
        यदि (वाक्यं_तुलय(कुञ्चिका, gguf_get_कुञ्चिका(ctx, i)) == 0) {
            keyfound = i;
            भङ्ग;
        }
    }

    प्रत्यागम keyfound;
}

ध्रुव वर्ण * gguf_get_कुञ्चिका(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_कुञ्चिका().c_str();
}

गणना gguf_type gguf_get_kv_type(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    प्रत्यागम ctx->kv[कुञ्चिका_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[कुञ्चिका_id].get_type();
}

गणना gguf_type gguf_get_arr_type(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].is_array);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_type();
}

ध्रुव शून्य * gguf_get_arr_दत्त(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_type() != GGUF_TYPE_STRING);
    प्रत्यागम ctx->kv[कुञ्चिका_id].दत्त.दत्त();
}

ध्रुव वर्ण * gguf_get_arr_str(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id, size_t i) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_type() == GGUF_TYPE_STRING);
    प्रत्यागम ctx->kv[कुञ्चिका_id].दत्त_string[i].c_str();
}

size_t gguf_get_arr_n(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));

    यदि (ctx->kv[कुञ्चिका_id].type == GGUF_TYPE_STRING) {
        प्रत्यागम ctx->kv[कुञ्चिका_id].दत्त_string.size();
    }

    ध्रुव size_t type_आकार = gguf_type_आकार(ctx->kv[कुञ्चिका_id].type);
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].दत्त.size() % type_आकार == 0);
    प्रत्यागम ctx->kv[कुञ्चिका_id].दत्त.size() / type_आकार;
}

uint8_t gguf_get_val_u8(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<int32_t>();
}

भिन्न gguf_get_val_f32(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<भिन्न>();
}

uint64_t gguf_get_val_u64(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<int64_t>();
}

द्विगुण gguf_get_val_f64(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<द्विगुण>();
}

तार्किक gguf_get_val_bool(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<तार्किक>();
}

ध्रुव वर्ण * gguf_get_val_str(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    प्रत्यागम ctx->kv[कुञ्चिका_id].get_val<std::string>().c_str();
}

ध्रुव शून्य * gguf_get_val_दत्त(ध्रुव संरचना gguf_context * ctx, int64_t कुञ्चिका_id) {
    GGML_ASSERT(कुञ्चिका_id >= 0 && कुञ्चिका_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[कुञ्चिका_id].get_type() != GGUF_TYPE_STRING);
    प्रत्यागम ctx->kv[कुञ्चिका_id].दत्त.दत्त();
}

int64_t gguf_get_n_tensors(ध्रुव संरचना gguf_context * ctx) {
    प्रत्यागम ctx->info.size();
}

int64_t gguf_find_tensor(ध्रुव संरचना gguf_context * ctx, ध्रुव वर्ण * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ध्रुव int64_t n_tensors = gguf_get_n_tensors(ctx);

    प्रति (int64_t i = 0; i < n_tensors; ++i) {
        यदि (वाक्यं_तुलय(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            भङ्ग;
        }
    }

    प्रत्यागम tensor_id;
}

size_t gguf_get_tensor_विस्थापनम्(ध्रुव संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    प्रत्यागम ctx->info[tensor_id].विस्थापनम्;
}

ध्रुव वर्ण * gguf_get_tensor_name(ध्रुव संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    प्रत्यागम ctx->info[tensor_id].t.name;
}

ध्रुव int64_t * gguf_get_tensor_ne(ध्रुव संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    प्रत्यागम ctx->info[tensor_id].t.ne;
}

गणना ggml_type gguf_get_tensor_type(ध्रुव संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    प्रत्यागम ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_आकार(ध्रुव संरचना gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    प्रत्यागम ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_कुञ्चिका(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका) {
    ध्रुव int64_t कुञ्चिका_id = gguf_find_कुञ्चिका(ctx, कुञ्चिका);
    यदि (कुञ्चिका_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + कुञ्चिका_id);
    }
    प्रत्यागम कुञ्चिका_id;
}

प्रतिरूपम्<प्रकारनाम T>
स्थिर शून्य gguf_check_reserved_कुञ्चिकाः(ध्रुव std::string & कुञ्चिका, ध्रुव T val) {
    यदि (कुञ्चिका == GGUF_कुञ्चिका_GENERAL_ALIGNMENT) {
        यदि constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_कुञ्चिका_GENERAL_ALIGNMENT " must be power of 2");
        } अन्यथा {
            GGML_अप्रयुक्तम्(val);
            GGML_ABORT(GGUF_कुञ्चिका_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

शून्य gguf_समुच्चयः_val_u8(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, uint8_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_i8(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, int8_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_u16(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, uint16_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_i16(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, int16_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_u32(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, uint32_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_i32(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, int32_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_f32(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, भिन्न val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_u64(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, uint64_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_i64(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, int64_t val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_f64(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, द्विगुण val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_bool(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, तार्किक val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, val);
}

शून्य gguf_समुच्चयः_val_str(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, ध्रुव वर्ण * val) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, val);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);
    ctx->kv.emplace_back(कुञ्चिका, std::string(val));
}

शून्य gguf_समुच्चयः_arr_दत्त(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, गणना gguf_type type, ध्रुव शून्य * दत्त, size_t n) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, दत्त);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);

    ध्रुव size_t nbytes = n*gguf_type_आकार(type);
    std::vector<int8_t> tmp(nbytes);
    यदि (!tmp.रिक्तानि()) {
        स्मृतिं_प्रतिलिख(tmp.दत्त(), दत्त, nbytes);
    }
    ctx->kv.emplace_back(कुञ्चिका, tmp);
    ctx->kv.back().cast(type);
}

शून्य gguf_समुच्चयः_arr_str(संरचना gguf_context * ctx, ध्रुव वर्ण * कुञ्चिका, ध्रुव वर्ण ** दत्त, size_t n) {
    gguf_check_reserved_कुञ्चिकाः(कुञ्चिका, दत्त);
    gguf_remove_कुञ्चिका(ctx, कुञ्चिका);

    std::vector<std::string> tmp(n);
    प्रति (size_t i = 0; i < n; ++i) {
        tmp[i] = दत्त[i];
    }
    ctx->kv.emplace_back(कुञ्चिका, tmp);
}

// set or add KV pairs from another context
शून्य gguf_समुच्चयः_kv(संरचना gguf_context * ctx, ध्रुव संरचना gguf_context * src) {
    ध्रुव int64_t n_kv = gguf_get_n_kv(src);
    प्रति (int64_t i = 0; i < n_kv; ++i) {
        ध्रुव संरचना gguf_kv & kv = src->kv[i];

        यदि (!kv.is_array) {
            विकल्प (kv.get_type()) {
                पक्ष GGUF_TYPE_UINT8:   gguf_समुच्चयः_val_u8  (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<uint8_t>());             भङ्ग;
                पक्ष GGUF_TYPE_INT8:    gguf_समुच्चयः_val_i8  (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<int8_t>());              भङ्ग;
                पक्ष GGUF_TYPE_UINT16:  gguf_समुच्चयः_val_u16 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<uint16_t>());            भङ्ग;
                पक्ष GGUF_TYPE_INT16:   gguf_समुच्चयः_val_i16 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<int16_t>());             भङ्ग;
                पक्ष GGUF_TYPE_UINT32:  gguf_समुच्चयः_val_u32 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<uint32_t>());            भङ्ग;
                पक्ष GGUF_TYPE_INT32:   gguf_समुच्चयः_val_i32 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<int32_t>());             भङ्ग;
                पक्ष GGUF_TYPE_FLOAT32: gguf_समुच्चयः_val_f32 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<भिन्न>());               भङ्ग;
                पक्ष GGUF_TYPE_UINT64:  gguf_समुच्चयः_val_u64 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<uint64_t>());            भङ्ग;
                पक्ष GGUF_TYPE_INT64:   gguf_समुच्चयः_val_i64 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<int64_t>());             भङ्ग;
                पक्ष GGUF_TYPE_FLOAT64: gguf_समुच्चयः_val_f64 (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<द्विगुण>());              भङ्ग;
                पक्ष GGUF_TYPE_BOOL:    gguf_समुच्चयः_val_bool(ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<तार्किक>());                भङ्ग;
                पक्ष GGUF_TYPE_STRING:  gguf_समुच्चयः_val_str (ctx, kv.get_कुञ्चिका().c_str(), kv.get_val<std::string>().c_str()); भङ्ग;
                पक्ष GGUF_TYPE_ARRAY:
                स्वभाव: GGML_ABORT("invalid type");
            }
            अनुवर्त;
        }

        ध्रुव size_t ne = kv.get_ne();

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
                gguf_समुच्चयः_arr_दत्त(ctx, kv.get_कुञ्चिका().c_str(), kv.get_type(), kv.दत्त.दत्त(), ne);
            } भङ्ग;
            पक्ष GGUF_TYPE_STRING: {
                std::vector<ध्रुव वर्ण *> tmp(ne);
                प्रति (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.दत्त_string[j].c_str();
                }
                gguf_समुच्चयः_arr_str(ctx, kv.get_कुञ्चिका().c_str(), tmp.दत्त(), ne);
            } भङ्ग;
            पक्ष GGUF_TYPE_ARRAY:
            स्वभाव: GGML_ABORT("invalid type");
        }
    }
}

शून्य gguf_सङ्कलय_tensor(
             संरचना gguf_context * ctx,
        ध्रुव संरचना ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    यदि (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    संरचना gguf_tensor_info ti;
    ti.t = *tensor;
    ti.विस्थापनम् = ctx->info.रिक्तानि() ? 0 :
        ctx->info.back().विस्थापनम् + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

शून्य gguf_समुच्चयः_tensor_type(संरचना gguf_context * ctx, ध्रुव वर्ण * name, गणना ggml_type type) {
    ध्रुव int64_t tensor_id = gguf_find_tensor(ctx, name);
    यदि (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    संरचना ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ध्रुव size_t  type_आकार = ggml_type_आकार(type);
    ध्रुव int64_t blck_आकार = ggml_blck_आकार(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_आकार == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_आकार;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_आकार);
    प्रति (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ध्रुव int64_t n_tensors = gguf_get_n_tensors(ctx);
    प्रति (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].विस्थापनम् = ctx->info[i - 1].विस्थापनम् + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

शून्य gguf_समुच्चयः_tensor_दत्त(संरचना gguf_context * ctx, ध्रुव वर्ण * name, ध्रुव शून्य * दत्त) {
    ध्रुव int64_t tensor_id = gguf_find_tensor(ctx, name);
    यदि (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.दत्त = (शून्य *)(uintptr_t)दत्त; // double cast suppresses warning about casting away const
}

संरचना gguf_writer_आधारः {
    size_t written_अष्टकानि {0u};

    आभास ~gguf_writer_आधारः(शून्य) = स्वभाव;

    // we bet on devirtualization
    आभास शून्य लिख(int8_t val) = 0;
    आभास शून्य लिख(ध्रुव std::vector<int8_t> & val) = 0;
    आभास शून्य write_tensor_दत्त(ध्रुव संरचना gguf_tensor_info & info, size_t विस्थापनम्_दत्त, size_t alignment) = 0;

    प्रतिरूपम् <प्रकारनाम T>
    शून्य लिख(ध्रुव T & val) {
        प्रति (size_t i = 0; i < परिमाण(val); ++i) {
            लिख(reinterpret_cast<ध्रुव int8_t *>(&val)[i]);
        }
    }

    शून्य लिख(ध्रुव तार्किक & val) {
        ध्रुव int8_t val8 = val ? 1 : 0;
        लिख(val8);
    }

    शून्य लिख(ध्रुव std::string & val) {
        {
            ध्रुव uint64_t n = val.दीर्घता();
            लिख(n);
        }
        प्रति (size_t i = 0; i < val.दीर्घता(); ++i) {
            लिख((val.दत्त())[i]);
        }
    }

    शून्य लिख(ध्रुव वर्ण * val) {
        लिख(std::string(val));
    }

    शून्य लिख(ध्रुव गणना ggml_type & val) {
        लिख(int32_t(val));
    }

    शून्य लिख(ध्रुव गणना gguf_type & val) {
        लिख(int32_t(val));
    }

    शून्य लिख(ध्रुव संरचना gguf_kv & kv) {
        ध्रुव uint64_t ne = kv.get_ne();

        लिख(kv.get_कुञ्चिका());

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
                लिख(kv.दत्त);
            } भङ्ग;
            पक्ष GGUF_TYPE_BOOL: {
                प्रति (size_t i = 0; i < ne; ++i) {
                    लिख(kv.get_val<तार्किक>(i));
                }
            } भङ्ग;
            पक्ष GGUF_TYPE_STRING: {
                प्रति (size_t i = 0; i < ne; ++i) {
                    लिख(kv.get_val<std::string>(i));
                }
            } भङ्ग;
            पक्ष GGUF_TYPE_ARRAY:
            स्वभाव: GGML_ABORT("invalid type");
        }
    }

    शून्य write_tensor_meta(ध्रुव संरचना gguf_tensor_info & info) {
        लिख(info.t.name);

        ध्रुव uint32_t n_dims = ggml_n_dims(&info.t);
        लिख(n_dims);

        प्रति (uint32_t j = 0; j < n_dims; ++j) {
            लिख(info.t.ne[j]);
        }
        लिख(info.t.type);
        लिख(info.विस्थापनम्);
    }

    शून्य pad(ध्रुव size_t alignment) {
        यावत् (written_अष्टकानि % alignment != 0) {
            ध्रुव int8_t zero = 0;
            लिख(zero);
        }
    }
};

// vector buffer based writer
संरचना gguf_writer_buf final : सार्वजनिक gguf_writer_आधारः {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    उपयुज्य gguf_writer_आधारः::लिख;

    शून्य लिख(ध्रुव int8_t val) override {
        buf.push_back(val);
        written_अष्टकानि++;
    }

    शून्य लिख(ध्रुव std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_अष्टकानि += val.size();
    }

    शून्य write_tensor_दत्त(ध्रुव संरचना gguf_tensor_info & info, ध्रुव size_t विस्थापनम्_दत्त, ध्रुव size_t alignment) override {
        GGML_ASSERT(buf.size() - विस्थापनम्_दत्त == info.विस्थापनम्);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ध्रुव size_t विस्थापनम् = buf.size();
        ध्रुव size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(विस्थापनम् + nbytes);
        यदि (info.t.बफर) {
            ggml_backend_tensor_get(&info.t, buf.दत्त() + विस्थापनम्, 0, nbytes);
        } अन्यथा {
            GGML_ASSERT(info.t.दत्त);
            स्मृतिं_प्रतिलिख(buf.दत्त() + विस्थापनम्, info.t.दत्त, nbytes);
        }
        written_अष्टकानि += nbytes;

        pad(alignment);
    }
};

// file based writer
संरचना gguf_writer_file final : सार्वजनिक gguf_writer_आधारः {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    उपयुज्य gguf_writer_आधारः::लिख;

    शून्य लिख(ध्रुव int8_t val) override {
        ध्रुव स्वचालित real_val = static_cast<uint8_t>(val);
        ध्रुव स्वचालित ret = fputc(real_val, file);
        written_अष्टकानि++;
        यदि (ret != real_val) {
            क्षिप std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((पूर्णाङ्क)real_val) + "'");
        }
    }

    शून्य लिख(ध्रुव std::vector<int8_t> & val) override {
        ध्रुव स्वचालित ret = fwrite(val.दत्त(), 1, val.size(), file);
        written_अष्टकानि += val.size();
        यदि (ret != val.size()) {
            क्षिप std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    शून्य write_tensor_दत्त(ध्रुव संरचना gguf_tensor_info & info, ध्रुव size_t विस्थापनम्_दत्त, ध्रुव size_t alignment) override {
        GGML_ASSERT(written_अष्टकानि - विस्थापनम्_दत्त == info.विस्थापनम्);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ध्रुव size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        यदि (info.t.बफर) {
            ggml_backend_tensor_get(&info.t, buf.दत्त(), 0, nbytes);
        } अन्यथा {
            GGML_ASSERT(info.t.दत्त);
            स्मृतिं_प्रतिलिख(buf.दत्त(), info.t.दत्त, nbytes);
        }
        लिख(buf);

        pad(alignment);
    }
};

प्रतिरूपम् <प्रकारनाम writer_t>
स्थिर शून्य gguf_write_बहिर्गमनम्(ध्रुव संरचना gguf_context * ctx, writer_t & gw, तार्किक only_meta) {
    ध्रुव int64_t n_kv      = gguf_get_n_kv(ctx);
    ध्रुव int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.लिख(GGUF_MAGIC[0]);
    gw.लिख(GGUF_MAGIC[1]);
    gw.लिख(GGUF_MAGIC[2]);
    gw.लिख(GGUF_MAGIC[3]);
    gw.लिख(ctx->संस्करणम्_2);
    gw.लिख(n_tensors);
    gw.लिख(n_kv);

    // write key-value pairs
    प्रति (int64_t i = 0; i < n_kv; ++i) {
        gw.लिख(ctx->kv[i]);
    }

    // write tensor info
    प्रति (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    यदि (only_meta) {
        प्रत्यागम;
    }

    ध्रुव size_t विस्थापनम्_दत्त = gw.written_अष्टकानि;

    // write tensor data
    प्रति (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_दत्त(ctx->info[i], विस्थापनम्_दत्त, ctx->alignment);
    }
}

शून्य gguf_write_गन्तव्यम्_buf(ध्रुव संरचना gguf_context * ctx, std::vector<int8_t> & buf, तार्किक only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_बहिर्गमनम्(ctx, gw, only_meta);
}

तार्किक gguf_write_गन्तव्यम्_file_ptr(ध्रुव संरचना gguf_context * ctx, FILE * file, तार्किक only_meta) {
    GGML_ASSERT(file);

    यत्न {
        gguf_writer_file gw(file);
        gguf_write_बहिर्गमनम्(ctx, gw, only_meta);
    } गृहाण (ध्रुव std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

तार्किक gguf_write_गन्तव्यम्_file(ध्रुव संरचना gguf_context * ctx, ध्रुव वर्ण * fname, तार्किक only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    यदि (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        प्रत्यागम असत्य;
    }

    ध्रुव तार्किक success = gguf_write_गन्तव्यम्_file_ptr(ctx, file, only_meta);
    यदि (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    प्रत्यागम success;
}

size_t gguf_get_meta_आकार(ध्रुव संरचना gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_गन्तव्यम्_buf(ctx, buf, /*only_meta =*/ सत्य);
    प्रत्यागम buf.size();
}

शून्य gguf_get_meta_दत्त(ध्रुव संरचना gguf_context * ctx, शून्य * दत्त) {
    std::vector<int8_t> buf;
    gguf_write_गन्तव्यम्_buf(ctx, buf, /*only_meta =*/ सत्य);
    स्मृतिं_प्रतिलिख(दत्त, buf.दत्त(), buf.size());
}
