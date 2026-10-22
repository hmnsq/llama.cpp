#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "gguf.h"

#അന്തര്ഭാവ <cinttypes>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <cerrno>
#അന്തര്ഭാവ <map>
#അന്തര്ഭാവ <new>
#അന്തര്ഭാവ <stdexcept>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#അല്ലെങ്കിൽ
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
ഘടന type_to_gguf_type;

template <>
ഘടന type_to_gguf_type<uint8_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_UINT8;
};

template <>
ഘടന type_to_gguf_type<int8_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_INT8;
};

template <>
ഘടന type_to_gguf_type<uint16_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_UINT16;
};

template <>
ഘടന type_to_gguf_type<int16_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_INT16;
};

template <>
ഘടന type_to_gguf_type<uint32_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_UINT32;
};

template <>
ഘടന type_to_gguf_type<int32_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_INT32;
};

template <>
ഘടന type_to_gguf_type<ദശാംശം> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
ഘടന type_to_gguf_type<തർക്കം> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_BOOL;
};

template <>
ഘടന type_to_gguf_type<std::string> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_STRING;
};

template <>
ഘടന type_to_gguf_type<uint64_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_UINT64;
};

template <>
ഘടന type_to_gguf_type<int64_t> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_INT64;
};

template <>
ഘടന type_to_gguf_type<ഇരട്ട> {
    സ്ഥിതം constexpr എണ്ണൽ gguf_type value = GGUF_TYPE_FLOAT64;
};

സ്ഥിതം സ്ഥിരം std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   വലുപ്പം(uint8_t)},
    {GGUF_TYPE_INT8,    വലുപ്പം(int8_t)},
    {GGUF_TYPE_UINT16,  വലുപ്പം(uint16_t)},
    {GGUF_TYPE_INT16,   വലുപ്പം(int16_t)},
    {GGUF_TYPE_UINT32,  വലുപ്പം(uint32_t)},
    {GGUF_TYPE_INT32,   വലുപ്പം(int32_t)},
    {GGUF_TYPE_FLOAT32, വലുപ്പം(ദശാംശം)},
    {GGUF_TYPE_BOOL,    വലുപ്പം(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  വലുപ്പം(uint64_t)},
    {GGUF_TYPE_INT64,   വലുപ്പം(int64_t)},
    {GGUF_TYPE_FLOAT64, വലുപ്പം(ഇരട്ട)},
};
static_assert(GGUF_TYPE_എണ്ണം == 13, "GGUF_TYPE_COUNT != 13");

സ്ഥിതം സ്ഥിരം std::map<gguf_type, സ്ഥിരം അക്ഷരം *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_എണ്ണം == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(എണ്ണൽ gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    തിരികെ it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

ഘടന gguf_kv {
    std::string താക്കോൽ;

    തർക്കം is_array;
    എണ്ണൽ gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(സ്ഥിരം std::string & താക്കോൽ, സ്ഥിരം T value)
            : താക്കോൽ(താക്കോൽ), is_array(അസത്യം), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!താക്കോൽ.empty());
        data.resize(വലുപ്പം(T));
        memcpy(data.data(), &value, വലുപ്പം(T));
    }

    template <typename T>
    gguf_kv(സ്ഥിരം std::string & താക്കോൽ, സ്ഥിരം std::vector<T> & value)
            : താക്കോൽ(താക്കോൽ), is_array(സത്യം), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!താക്കോൽ.empty());
        data.resize(value.size()*വലുപ്പം(T));
        വേണ്ടി (size_t i = 0; i < value.size(); ++i) {
            സ്ഥിരം T tmp = value[i];
            memcpy(data.data() + i*വലുപ്പം(T), &tmp, വലുപ്പം(T));
        }
    }

    gguf_kv(സ്ഥിരം std::string & താക്കോൽ, സ്ഥിരം std::string & value)
            : താക്കോൽ(താക്കോൽ), is_array(അസത്യം), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!താക്കോൽ.empty());
        data_string.push_back(value);
    }

    gguf_kv(സ്ഥിരം std::string & താക്കോൽ, സ്ഥിരം std::vector<std::string> & value)
            : താക്കോൽ(താക്കോൽ), is_array(സത്യം), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!താക്കോൽ.empty());
        data_string = value;
    }

    സ്ഥിരം std::string & get_താക്കോൽ() സ്ഥിരം {
        തിരികെ താക്കോൽ;
    }

    സ്ഥിരം എണ്ണൽ gguf_type & get_type() സ്ഥിരം {
        തിരികെ type;
    }

    size_t get_ne() സ്ഥിരം {
        എങ്കിൽ (type == GGUF_TYPE_STRING) {
            സ്ഥിരം size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            തിരികെ ne;
        }
        സ്ഥിരം size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        സ്ഥിരം size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        തിരികെ ne;
    }

    template <typename T>
    സ്ഥിരം T & get_val(സ്ഥിരം size_t i = 0) സ്ഥിരം {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        എങ്കിൽ constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            തിരികെ data_string[i];
        }
        സ്ഥിരം size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        തിരികെ reinterpret_cast<സ്ഥിരം T *>(data.data())[i];
    }

    ശൂന്യം cast(സ്ഥിരം എണ്ണൽ gguf_type new_type) {
        സ്ഥിരം size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

ഘടന gguf_tensor_info {
    ഘടന ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

ഘടന gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<ഘടന gguf_kv> kv;
    std::vector<ഘടന gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    ശൂന്യം * data = nullptr;
};

ഘടന gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            ശൂന്യം * userdata,
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
    സ്ഥിതം uint64_t file_remain(FILE * file) {
        സ്ഥിരം int64_t cur = gguf_ftell(file);
        എങ്കിൽ (cur < 0) {
            തിരികെ 0;
        }
        എങ്കിൽ (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            തിരികെ 0;
        }
        സ്ഥിരം int64_t end = gguf_ftell(file);
        എങ്കിൽ (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            തിരികെ 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        തിരികെ static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    തർക്കം പഠ(T & dst) സ്ഥിരം {
        സ്ഥിരം size_t size = വലുപ്പം(dst);
        എങ്കിൽ (size > nbytes_remain) {
            തിരികെ അസത്യം;
        }
        തിരികെ read_raw(&dst, size) == size;
    }

    template <typename T>
    തർക്കം പഠ(std::vector<T> & dst, സ്ഥിരം size_t n) സ്ഥിരം {
        എങ്കിൽ (n > GGUF_MAX_ARRAY_ELEMENTS) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            എങ്കിൽ (n > SIZE_MAX / വലുപ്പം(uint64_t)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (nbytes_remain < n * വലുപ്പം(uint64_t)) {
                തിരികെ അസത്യം;
            }
        } അല്ലെങ്കിൽ {
            എങ്കിൽ (n > SIZE_MAX / വലുപ്പം(T)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (nbytes_remain < n * വലുപ്പം(T)) {
                തിരികെ അസത്യം;
            }
        }
        dst.resize(n);
        വേണ്ടി (size_t i = 0; i < dst.size(); ++i) {
            എങ്കിൽ constexpr (std::is_same<T, തർക്കം>::value) {
                തർക്കം tmp;
                എങ്കിൽ (!പഠ(tmp)) {
                    തിരികെ അസത്യം;
                }
                dst[i] = tmp;
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (!പഠ(dst[i])) {
                    തിരികെ അസത്യം;
                }
            }
        }
        തിരികെ സത്യം;
    }

    തർക്കം പഠ(തർക്കം & dst) സ്ഥിരം {
        int8_t tmp = -1;
        എങ്കിൽ (!പഠ(tmp)) {
            തിരികെ അസത്യം;
        }
        dst = tmp != 0;
        തിരികെ സത്യം;
    }

    തർക്കം പഠ(എണ്ണൽ ggml_type & dst) സ്ഥിരം {
        int32_t tmp = -1;
        എങ്കിൽ (!പഠ(tmp)) {
            തിരികെ അസത്യം;
        }
        dst = ggml_type(tmp);
        തിരികെ സത്യം;
    }

    തർക്കം പഠ(എണ്ണൽ gguf_type & dst) സ്ഥിരം {
        int32_t tmp = -1;
        എങ്കിൽ (!പഠ(tmp)) {
            തിരികെ അസത്യം;
        }
        dst = gguf_type(tmp);
        തിരികെ സത്യം;
    }

    തർക്കം പഠ(std::string & dst) സ്ഥിരം {
        uint64_t size = 0;
        എങ്കിൽ (!പഠ(size)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            തിരികെ അസത്യം;
        }
        dst.resize(static_cast<size_t>(size));
        തിരികെ read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    തർക്കം പഠ(ശൂന്യം * dst, സ്ഥിരം size_t size) സ്ഥിരം {
        എങ്കിൽ (size > nbytes_remain) {
            തിരികെ അസത്യം;
        }
        തിരികെ read_raw(dst, size) == size;
    }

    uint64_t tell() സ്ഥിരം {
        തിരികെ data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() സ്ഥിരം {
        തിരികെ start_offset;
    }

    തർക്കം seek(uint64_t absolute_offset) സ്ഥിരം {
        സ്ഥിരം uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        എങ്കിൽ (absolute_offset > end_offset) {
            തിരികെ അസത്യം;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        തിരികെ സത്യം;
    }

സ്വകാര്യ:
    size_t read_raw(ശൂന്യം * dst, size_t size) സ്ഥിരം {
        എങ്കിൽ (callback == nullptr || size == 0) {
            തിരികെ 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        തർക്കം reached_eof = അസത്യം;

        ആകുമ്പോൾ (total_nread < size) {
            സ്ഥിരം size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            എങ്കിൽ (data_offset + total_nread < data_offset) {
                നിർത്തുക;
            }
            സ്ഥിരം size_t nread = callback(userdata, static_cast<ശൂന്യം *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            എങ്കിൽ (nread != chunk_size) {
                reached_eof = സത്യം;
                നിർത്തുക;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        എങ്കിൽ (reached_eof) {
            nbytes_remain = 0;
        }

        തിരികെ total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    ശൂന്യം * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

ഘടന gguf_context * gguf_init_empty(ശൂന്യം) {
    തിരികെ പുതിയ gguf_context;
}

template<typename T>
തർക്കം gguf_read_emplace_helper(സ്ഥിരം ഘടന gguf_reader & gr, std::vector<ഘടന gguf_kv> & kv, സ്ഥിരം std::string & താക്കോൽ, സ്ഥിരം തർക്കം is_array, സ്ഥിരം size_t n) {
    എങ്കിൽ (is_array) {
        std::vector<T> value;
        ശ്രമിക്കുക {
            എങ്കിൽ (!gr.പഠ(value, n)) {
                തിരികെ അസത്യം;
            }
        } പിടിക്കുക (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, താക്കോൽ.c_str());
            തിരികെ അസത്യം;
        } പിടിക്കുക (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, താക്കോൽ.c_str());
            തിരികെ അസത്യം;
        }
        kv.emplace_back(താക്കോൽ, value);
    } അല്ലെങ്കിൽ {
        T value;
        എങ്കിൽ (!gr.പഠ(value)) {
            തിരികെ അസത്യം;
        }
        kv.emplace_back(താക്കോൽ, value);
    }
    തിരികെ സത്യം;
}

സ്ഥിതം ഘടന gguf_context * gguf_init_from_reader(സ്ഥിരം ഘടന gguf_reader & gr, ഘടന gguf_init_params params) {
    ഘടന gguf_context * ctx = പുതിയ gguf_context;

    തർക്കം ok = സത്യം;

    // file magic
    {
        std::vector<അക്ഷരം> magic;
        ok = ok && gr.പഠ(magic, 4);

        എങ്കിൽ (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            തിരികെ nullptr;
        }

        വേണ്ടി (uint32_t i = 0; i < magic.size(); i++) {
            എങ്കിൽ (magic[i] != GGUF_MAGIC[i]) {
                അക്ഷരം c0 = isprint(magic[0]) ? magic[0] : '?';
                അക്ഷരം c1 = isprint(magic[1]) ? magic[1] : '?';
                അക്ഷരം c2 = isprint(magic[2]) ? magic[2] : '?';
                അക്ഷരം c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                തിരികെ nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    എങ്കിൽ (ok && gr.പഠ(ctx->version)) {
        എങ്കിൽ (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = അസത്യം;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        എങ്കിൽ (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = അസത്യം;
        }

        എങ്കിൽ (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = അസത്യം;
        }
        എങ്കിൽ (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        ok = അസത്യം;
    }

    എങ്കിൽ (ok && gr.പഠ(n_tensors)) {
        static_assert(വലുപ്പം(size_t) <= 8 && വലുപ്പം(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        എങ്കിൽ (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/വലുപ്പം(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/വലുപ്പം(gguf_tensor_info));
            ok = അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        ok = അസത്യം;
    }

    എങ്കിൽ (ok && gr.പഠ(n_kv)) {
        static_assert(വലുപ്പം(size_t) <= 8 && വലുപ്പം(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        എങ്കിൽ (n_kv < 0 || n_kv > int64_t(SIZE_MAX/വലുപ്പം(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/വലുപ്പം(gguf_kv));
            ok = അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        ok = അസത്യം;
    }

    എങ്കിൽ (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        തിരികെ nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_താക്കോലുകൾ;

        വേണ്ടി (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string താക്കോൽ;
            gguf_type   type     = gguf_type(-1);
            തർക്കം        is_array = അസത്യം;
            uint64_t    n        = 1;

            ശ്രമിക്കുക {
                ok = ok && gr.പഠ(താക്കോൽ);
            } പിടിക്കുക (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = അസത്യം;
            } പിടിക്കുക (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = അസത്യം;
            }
            എങ്കിൽ (ok && താക്കോൽ.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = അസത്യം;
            }
            എങ്കിൽ (ok && !seen_താക്കോലുകൾ.insert(താക്കോൽ).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, താക്കോൽ.c_str(), i);
                ok = അസത്യം;
            }
            എങ്കിൽ (!ok) {
                നിർത്തുക;
            }

            ok = ok && gr.പഠ(type);
            എങ്കിൽ (type == GGUF_TYPE_ARRAY) {
                is_array = സത്യം;
                ok = ok && gr.പഠ(type);
                ok = ok && gr.പഠ(n);
            }
            എങ്കിൽ (!ok) {
                നിർത്തുക;
            }

            തിരഞ്ഞെടുപ്പ് (type) {
                അവസ്ഥ GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<ദശാംശം>      (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<തർക്കം>       (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<ഇരട്ട>     (gr, ctx->kv, താക്കോൽ, is_array, n); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_ARRAY:
                സ്ഥിരസ്ഥിതി:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, താക്കോൽ.c_str(), type);
                        ok = അസത്യം;
                    } നിർത്തുക;
            }
        }

        എങ്കിൽ (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            തിരികെ nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        സ്ഥിരം പൂർണ്ണസംഖ്യ alignment_idx = gguf_find_താക്കോൽ(ctx, GGUF_താക്കോൽ_GENERAL_ALIGNMENT);
        എങ്കിൽ (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_താക്കോൽ_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            തിരികെ nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        എങ്കിൽ (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            തിരികെ nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    വേണ്ടി (int64_t i = 0; ok && i < n_tensors; ++i) {
        ഘടന gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            ശ്രമിക്കുക {
                ok = ok && gr.പഠ(name);
            } പിടിക്കുക (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = അസത്യം;
            } പിടിക്കുക (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = അസത്യം;
            }
            എങ്കിൽ (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = അസത്യം;
                നിർത്തുക;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            എങ്കിൽ (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (!ok) {
            നിർത്തുക;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.പഠ(n_dims);
            എങ്കിൽ (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = അസത്യം;
                നിർത്തുക;
            }
            വേണ്ടി (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                എങ്കിൽ (j < n_dims) {
                    ok = ok && gr.പഠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                എങ്കിൽ (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = അസത്യം;
                    നിർത്തുക;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            എങ്കിൽ (ok) {
                int64_t current_nelements = info.t.ne[0];
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < GGML_MAX_DIMS; ++i) {
                    എങ്കിൽ (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        എങ്കിൽ (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = അസത്യം;
                             നിർത്തുക;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 എങ്കിൽ (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     നിർത്തുക;
                 }
            }
        }
        എങ്കിൽ (!ok) {
            നിർത്തുക;
        }

        // tensor type
        {
            ok = ok && gr.പഠ(info.t.type);

            // check that tensor type is within defined range
            എങ്കിൽ (info.t.type < 0 || info.t.type >= GGML_TYPE_എണ്ണം) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_എണ്ണം);
                ok = അസത്യം;
                നിർത്തുക;
            }
            സ്ഥിരം size_t  type_size = ggml_type_size(info.t.type);
            സ്ഥിരം int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            എങ്കിൽ (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (പൂർണ്ണസംഖ്യ) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = അസത്യം;
                നിർത്തുക;
            }

            // check that the size of the tensor in bytes is representable
            എങ്കിൽ (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = അസത്യം;
                നിർത്തുക;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        എങ്കിൽ (!ok) {
            നിർത്തുക;
        }

        // tensor data offset within buffer
        ok = ok && gr.പഠ(info.offset);

        ctx->info.push_back(info);
    }

    എങ്കിൽ (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        തിരികെ nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    എങ്കിൽ (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        തിരികെ nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        വേണ്ടി (size_t i = 0; i < ctx->info.size(); ++i) {
            സ്ഥിരം gguf_tensor_info & ti = ctx->info[i];
            എങ്കിൽ (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                തിരികെ nullptr;
            }
            സ്ഥിരം size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            എങ്കിൽ (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                തിരികെ nullptr;
            }
            സ്ഥിരം size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            എങ്കിൽ (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                തിരികെ nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    എങ്കിൽ (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        എങ്കിൽ (params.no_alloc) {
            എങ്കിൽ (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                തിരികെ nullptr;
            }

            സ്ഥിരം size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } അല്ലെങ്കിൽ {
            എങ്കിൽ ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                തിരികെ nullptr;
            }

            സ്ഥിരം size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            എങ്കിൽ (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                തിരികെ nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        ഘടന ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        എങ്കിൽ (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            തിരികെ nullptr;
        }

        ഘടന ggml_context * ctx_data = *params.ctx;

        ഘടന ggml_tensor * data = nullptr;

        എങ്കിൽ (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            എങ്കിൽ (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.പഠ(data->data, ctx->size);

            എങ്കിൽ (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                തിരികെ nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, സത്യം);

        // create the tensors
        വേണ്ടി (size_t i = 0; i < ctx->info.size(); ++i) {
            സ്ഥിരം ഘടന gguf_tensor_info & info = ctx->info[i];

            ഘടന ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            എങ്കിൽ (!ok) {
                നിർത്തുക;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            എങ്കിൽ (!params.no_alloc) {
                cur->data = (അക്ഷരം *) data->data + info.offset;
            }
        }

        എങ്കിൽ (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            തിരികെ nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    തിരികെ ctx;
}

ഘടന gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, ശൂന്യം * userdata, size_t max_chunk_read, uint64_t max_expected_size, ഘടന gguf_init_params params) {
    എങ്കിൽ (callback == nullptr) {
        തിരികെ nullptr;
    }

    സ്ഥിരം ഘടന gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    തിരികെ gguf_init_from_reader(gr, params);
}

ഘടന gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

സ്ഥിതം size_t gguf_file_reader_callback(ശൂന്യം * userdata, ശൂന്യം * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    എങ്കിൽ (reader.offset != offset) {
        എങ്കിൽ (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            തിരികെ 0;
        }

        reader.offset = offset;
    }

    സ്ഥിരം size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    തിരികെ nread;
}

ഘടന gguf_context * gguf_init_from_file_ptr(FILE * file, ഘടന gguf_init_params params) {
    എങ്കിൽ (!file) {
        തിരികെ nullptr;
    }

    സ്ഥിരം int64_t cur = gguf_ftell(file);
    എങ്കിൽ (cur < 0) {
        തിരികെ nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    സ്ഥിരം ഘടന gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    തിരികെ gguf_init_from_reader(gr, params);
}

ഘടന gguf_buffer_reader {
    സ്ഥിരം uint8_t * data;
    size_t          size;
};

സ്ഥിതം size_t gguf_buffer_reader_callback(ശൂന്യം * userdata, ശൂന്യം * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    സ്ഥിരം gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    എങ്കിൽ (offset > reader.size || len > reader.size - offset) {
        തിരികെ 0;
    }

    സ്ഥിരം size_t data_offset = static_cast<size_t>(offset);
    സ്ഥിരം size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    തിരികെ nread;
}

ഘടന gguf_context * gguf_init_from_buffer(സ്ഥിരം ശൂന്യം * data, size_t size, ഘടന gguf_init_params params) {
    എങ്കിൽ (data == nullptr || size == 0) {
        തിരികെ nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<സ്ഥിരം uint8_t *>(data),
        /*.size = */ size,
    };
    സ്ഥിരം ഘടന gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    തിരികെ gguf_init_from_reader(gr, params);
}

ഘടന gguf_context * gguf_init_from_file(സ്ഥിരം അക്ഷരം * fname, ഘടന gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    എങ്കിൽ (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        തിരികെ nullptr;
    }

    ഘടന gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    തിരികെ result;
}

ശൂന്യം gguf_free(ഘടന gguf_context * ctx) {
    എങ്കിൽ (ctx == nullptr) {
        തിരികെ;
    }
    delete ctx;
}

സ്ഥിരം അക്ഷരം * gguf_type_name(എണ്ണൽ gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    തിരികെ it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(സ്ഥിരം ഘടന gguf_context * ctx) {
    തിരികെ ctx->version;
}

size_t gguf_get_alignment(സ്ഥിരം ഘടന gguf_context * ctx) {
    തിരികെ ctx->alignment;
}

size_t gguf_get_data_offset(സ്ഥിരം ഘടന gguf_context * ctx) {
    തിരികെ ctx->offset;
}

int64_t gguf_get_n_kv(സ്ഥിരം ഘടന gguf_context * ctx) {
    തിരികെ ctx->kv.size();
}

int64_t gguf_find_താക്കോൽ(സ്ഥിരം ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ) {
    // return -1 if key not found
    int64_t keyfound = -1;

    സ്ഥിരം int64_t n_kv = gguf_get_n_kv(ctx);

    വേണ്ടി (int64_t i = 0; i < n_kv; ++i) {
        എങ്കിൽ (strcmp(താക്കോൽ, gguf_get_താക്കോൽ(ctx, i)) == 0) {
            keyfound = i;
            നിർത്തുക;
        }
    }

    തിരികെ keyfound;
}

സ്ഥിരം അക്ഷരം * gguf_get_താക്കോൽ(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    തിരികെ ctx->kv[താക്കോൽ_id].get_താക്കോൽ().c_str();
}

എണ്ണൽ gguf_type gguf_get_kv_type(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    തിരികെ ctx->kv[താക്കോൽ_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[താക്കോൽ_id].get_type();
}

എണ്ണൽ gguf_type gguf_get_arr_type(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].is_array);
    തിരികെ ctx->kv[താക്കോൽ_id].get_type();
}

സ്ഥിരം ശൂന്യം * gguf_get_arr_data(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_type() != GGUF_TYPE_STRING);
    തിരികെ ctx->kv[താക്കോൽ_id].data.data();
}

സ്ഥിരം അക്ഷരം * gguf_get_arr_str(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id, size_t i) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_type() == GGUF_TYPE_STRING);
    തിരികെ ctx->kv[താക്കോൽ_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));

    എങ്കിൽ (ctx->kv[താക്കോൽ_id].type == GGUF_TYPE_STRING) {
        തിരികെ ctx->kv[താക്കോൽ_id].data_string.size();
    }

    സ്ഥിരം size_t type_size = gguf_type_size(ctx->kv[താക്കോൽ_id].type);
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].data.size() % type_size == 0);
    തിരികെ ctx->kv[താക്കോൽ_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<int32_t>();
}

ദശാംശം gguf_get_val_f32(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<ദശാംശം>();
}

uint64_t gguf_get_val_u64(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<int64_t>();
}

ഇരട്ട gguf_get_val_f64(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<ഇരട്ട>();
}

തർക്കം gguf_get_val_bool(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<തർക്കം>();
}

സ്ഥിരം അക്ഷരം * gguf_get_val_str(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    തിരികെ ctx->kv[താക്കോൽ_id].get_val<std::string>().c_str();
}

സ്ഥിരം ശൂന്യം * gguf_get_val_data(സ്ഥിരം ഘടന gguf_context * ctx, int64_t താക്കോൽ_id) {
    GGML_ASSERT(താക്കോൽ_id >= 0 && താക്കോൽ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[താക്കോൽ_id].get_type() != GGUF_TYPE_STRING);
    തിരികെ ctx->kv[താക്കോൽ_id].data.data();
}

int64_t gguf_get_n_tensors(സ്ഥിരം ഘടന gguf_context * ctx) {
    തിരികെ ctx->info.size();
}

int64_t gguf_find_tensor(സ്ഥിരം ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    സ്ഥിരം int64_t n_tensors = gguf_get_n_tensors(ctx);

    വേണ്ടി (int64_t i = 0; i < n_tensors; ++i) {
        എങ്കിൽ (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            നിർത്തുക;
        }
    }

    തിരികെ tensor_id;
}

size_t gguf_get_tensor_offset(സ്ഥിരം ഘടന gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    തിരികെ ctx->info[tensor_id].offset;
}

സ്ഥിരം അക്ഷരം * gguf_get_tensor_name(സ്ഥിരം ഘടന gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    തിരികെ ctx->info[tensor_id].t.name;
}

സ്ഥിരം int64_t * gguf_get_tensor_ne(സ്ഥിരം ഘടന gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    തിരികെ ctx->info[tensor_id].t.ne;
}

എണ്ണൽ ggml_type gguf_get_tensor_type(സ്ഥിരം ഘടന gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    തിരികെ ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(സ്ഥിരം ഘടന gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    തിരികെ ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_താക്കോൽ(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ) {
    സ്ഥിരം int64_t താക്കോൽ_id = gguf_find_താക്കോൽ(ctx, താക്കോൽ);
    എങ്കിൽ (താക്കോൽ_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + താക്കോൽ_id);
    }
    തിരികെ താക്കോൽ_id;
}

template<typename T>
സ്ഥിതം ശൂന്യം gguf_check_reserved_താക്കോലുകൾ(സ്ഥിരം std::string & താക്കോൽ, സ്ഥിരം T val) {
    എങ്കിൽ (താക്കോൽ == GGUF_താക്കോൽ_GENERAL_ALIGNMENT) {
        എങ്കിൽ constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_താക്കോൽ_GENERAL_ALIGNMENT " must be power of 2");
        } അല്ലെങ്കിൽ {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_താക്കോൽ_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

ശൂന്യം gguf_set_val_u8(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, uint8_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_i8(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, int8_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_u16(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, uint16_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_i16(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, int16_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_u32(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, uint32_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_i32(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, int32_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_f32(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, ദശാംശം val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_u64(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, uint64_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_i64(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, int64_t val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_f64(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, ഇരട്ട val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_bool(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, തർക്കം val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, val);
}

ശൂന്യം gguf_set_val_str(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, സ്ഥിരം അക്ഷരം * val) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, val);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);
    ctx->kv.emplace_back(താക്കോൽ, std::string(val));
}

ശൂന്യം gguf_set_arr_data(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, എണ്ണൽ gguf_type type, സ്ഥിരം ശൂന്യം * data, size_t n) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, data);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);

    സ്ഥിരം size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    എങ്കിൽ (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(താക്കോൽ, tmp);
    ctx->kv.back().cast(type);
}

ശൂന്യം gguf_set_arr_str(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * താക്കോൽ, സ്ഥിരം അക്ഷരം ** data, size_t n) {
    gguf_check_reserved_താക്കോലുകൾ(താക്കോൽ, data);
    gguf_remove_താക്കോൽ(ctx, താക്കോൽ);

    std::vector<std::string> tmp(n);
    വേണ്ടി (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(താക്കോൽ, tmp);
}

// set or add KV pairs from another context
ശൂന്യം gguf_set_kv(ഘടന gguf_context * ctx, സ്ഥിരം ഘടന gguf_context * src) {
    സ്ഥിരം int64_t n_kv = gguf_get_n_kv(src);
    വേണ്ടി (int64_t i = 0; i < n_kv; ++i) {
        സ്ഥിരം ഘടന gguf_kv & kv = src->kv[i];

        എങ്കിൽ (!kv.is_array) {
            തിരഞ്ഞെടുപ്പ് (kv.get_type()) {
                അവസ്ഥ GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<uint8_t>());             നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<int8_t>());              നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<uint16_t>());            നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<int16_t>());             നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<uint32_t>());            നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<int32_t>());             നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<ദശാംശം>());               നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<uint64_t>());            നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<int64_t>());             നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<ഇരട്ട>());              നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<തർക്കം>());                നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_താക്കോൽ().c_str(), kv.get_val<std::string>().c_str()); നിർത്തുക;
                അവസ്ഥ GGUF_TYPE_ARRAY:
                സ്ഥിരസ്ഥിതി: GGML_ABORT("invalid type");
            }
            തുടരുക;
        }

        സ്ഥിരം size_t ne = kv.get_ne();

        തിരഞ്ഞെടുപ്പ് (kv.get_type()) {
            അവസ്ഥ GGUF_TYPE_UINT8:
            അവസ്ഥ GGUF_TYPE_INT8:
            അവസ്ഥ GGUF_TYPE_UINT16:
            അവസ്ഥ GGUF_TYPE_INT16:
            അവസ്ഥ GGUF_TYPE_UINT32:
            അവസ്ഥ GGUF_TYPE_INT32:
            അവസ്ഥ GGUF_TYPE_FLOAT32:
            അവസ്ഥ GGUF_TYPE_UINT64:
            അവസ്ഥ GGUF_TYPE_INT64:
            അവസ്ഥ GGUF_TYPE_FLOAT64:
            അവസ്ഥ GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_താക്കോൽ().c_str(), kv.get_type(), kv.data.data(), ne);
            } നിർത്തുക;
            അവസ്ഥ GGUF_TYPE_STRING: {
                std::vector<സ്ഥിരം അക്ഷരം *> tmp(ne);
                വേണ്ടി (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_താക്കോൽ().c_str(), tmp.data(), ne);
            } നിർത്തുക;
            അവസ്ഥ GGUF_TYPE_ARRAY:
            സ്ഥിരസ്ഥിതി: GGML_ABORT("invalid type");
        }
    }
}

ശൂന്യം gguf_add_tensor(
             ഘടന gguf_context * ctx,
        സ്ഥിരം ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    എങ്കിൽ (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    ഘടന gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

ശൂന്യം gguf_set_tensor_type(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * name, എണ്ണൽ ggml_type type) {
    സ്ഥിരം int64_t tensor_id = gguf_find_tensor(ctx, name);
    എങ്കിൽ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    ഘടന ggml_tensor * tensor = &ctx->info[tensor_id].t;
    സ്ഥിരം size_t  type_size = ggml_type_size(type);
    സ്ഥിരം int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    സ്ഥിരം int64_t n_tensors = gguf_get_n_tensors(ctx);
    വേണ്ടി (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

ശൂന്യം gguf_set_tensor_data(ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * name, സ്ഥിരം ശൂന്യം * data) {
    സ്ഥിരം int64_t tensor_id = gguf_find_tensor(ctx, name);
    എങ്കിൽ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (ശൂന്യം *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

ഘടന gguf_writer_base {
    size_t written_bytes {0u};

    മായാ ~gguf_writer_base(ശൂന്യം) = സ്ഥിരസ്ഥിതി;

    // we bet on devirtualization
    മായാ ശൂന്യം ലിഖ(int8_t val) = 0;
    മായാ ശൂന്യം ലിഖ(സ്ഥിരം std::vector<int8_t> & val) = 0;
    മായാ ശൂന്യം write_tensor_data(സ്ഥിരം ഘടന gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    ശൂന്യം ലിഖ(സ്ഥിരം T & val) {
        വേണ്ടി (size_t i = 0; i < വലുപ്പം(val); ++i) {
            ലിഖ(reinterpret_cast<സ്ഥിരം int8_t *>(&val)[i]);
        }
    }

    ശൂന്യം ലിഖ(സ്ഥിരം തർക്കം & val) {
        സ്ഥിരം int8_t val8 = val ? 1 : 0;
        ലിഖ(val8);
    }

    ശൂന്യം ലിഖ(സ്ഥിരം std::string & val) {
        {
            സ്ഥിരം uint64_t n = val.length();
            ലിഖ(n);
        }
        വേണ്ടി (size_t i = 0; i < val.length(); ++i) {
            ലിഖ((val.data())[i]);
        }
    }

    ശൂന്യം ലിഖ(സ്ഥിരം അക്ഷരം * val) {
        ലിഖ(std::string(val));
    }

    ശൂന്യം ലിഖ(സ്ഥിരം എണ്ണൽ ggml_type & val) {
        ലിഖ(int32_t(val));
    }

    ശൂന്യം ലിഖ(സ്ഥിരം എണ്ണൽ gguf_type & val) {
        ലിഖ(int32_t(val));
    }

    ശൂന്യം ലിഖ(സ്ഥിരം ഘടന gguf_kv & kv) {
        സ്ഥിരം uint64_t ne = kv.get_ne();

        ലിഖ(kv.get_താക്കോൽ());

        എങ്കിൽ (kv.is_array) {
            ലിഖ(GGUF_TYPE_ARRAY);
            ലിഖ(kv.get_type());
            ലിഖ(ne);
        } അല്ലെങ്കിൽ {
            ലിഖ(kv.get_type());
        }

        തിരഞ്ഞെടുപ്പ് (kv.get_type()) {
            അവസ്ഥ GGUF_TYPE_UINT8:
            അവസ്ഥ GGUF_TYPE_INT8:
            അവസ്ഥ GGUF_TYPE_UINT16:
            അവസ്ഥ GGUF_TYPE_INT16:
            അവസ്ഥ GGUF_TYPE_UINT32:
            അവസ്ഥ GGUF_TYPE_INT32:
            അവസ്ഥ GGUF_TYPE_FLOAT32:
            അവസ്ഥ GGUF_TYPE_UINT64:
            അവസ്ഥ GGUF_TYPE_INT64:
            അവസ്ഥ GGUF_TYPE_FLOAT64: {
                ലിഖ(kv.data);
            } നിർത്തുക;
            അവസ്ഥ GGUF_TYPE_BOOL: {
                വേണ്ടി (size_t i = 0; i < ne; ++i) {
                    ലിഖ(kv.get_val<തർക്കം>(i));
                }
            } നിർത്തുക;
            അവസ്ഥ GGUF_TYPE_STRING: {
                വേണ്ടി (size_t i = 0; i < ne; ++i) {
                    ലിഖ(kv.get_val<std::string>(i));
                }
            } നിർത്തുക;
            അവസ്ഥ GGUF_TYPE_ARRAY:
            സ്ഥിരസ്ഥിതി: GGML_ABORT("invalid type");
        }
    }

    ശൂന്യം write_tensor_meta(സ്ഥിരം ഘടന gguf_tensor_info & info) {
        ലിഖ(info.t.name);

        സ്ഥിരം uint32_t n_dims = ggml_n_dims(&info.t);
        ലിഖ(n_dims);

        വേണ്ടി (uint32_t j = 0; j < n_dims; ++j) {
            ലിഖ(info.t.ne[j]);
        }
        ലിഖ(info.t.type);
        ലിഖ(info.offset);
    }

    ശൂന്യം pad(സ്ഥിരം size_t alignment) {
        ആകുമ്പോൾ (written_bytes % alignment != 0) {
            സ്ഥിരം int8_t zero = 0;
            ലിഖ(zero);
        }
    }
};

// vector buffer based writer
ഘടന gguf_writer_buf final : പൊതു gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::ലിഖ;

    ശൂന്യം ലിഖ(സ്ഥിരം int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    ശൂന്യം ലിഖ(സ്ഥിരം std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    ശൂന്യം write_tensor_data(സ്ഥിരം ഘടന gguf_tensor_info & info, സ്ഥിരം size_t offset_data, സ്ഥിരം size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        സ്ഥിരം size_t offset = buf.size();
        സ്ഥിരം size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        എങ്കിൽ (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } അല്ലെങ്കിൽ {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
ഘടന gguf_writer_file final : പൊതു gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::ലിഖ;

    ശൂന്യം ലിഖ(സ്ഥിരം int8_t val) override {
        സ്ഥിരം auto real_val = static_cast<uint8_t>(val);
        സ്ഥിരം auto ret = fputc(real_val, file);
        written_bytes++;
        എങ്കിൽ (ret != real_val) {
            എറിയുക std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((പൂർണ്ണസംഖ്യ)real_val) + "'");
        }
    }

    ശൂന്യം ലിഖ(സ്ഥിരം std::vector<int8_t> & val) override {
        സ്ഥിരം auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        എങ്കിൽ (ret != val.size()) {
            എറിയുക std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    ശൂന്യം write_tensor_data(സ്ഥിരം ഘടന gguf_tensor_info & info, സ്ഥിരം size_t offset_data, സ്ഥിരം size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        സ്ഥിരം size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        എങ്കിൽ (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } അല്ലെങ്കിൽ {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        ലിഖ(buf);

        pad(alignment);
    }
};

template <typename writer_t>
സ്ഥിതം ശൂന്യം gguf_write_out(സ്ഥിരം ഘടന gguf_context * ctx, writer_t & gw, തർക്കം only_meta) {
    സ്ഥിരം int64_t n_kv      = gguf_get_n_kv(ctx);
    സ്ഥിരം int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.ലിഖ(GGUF_MAGIC[0]);
    gw.ലിഖ(GGUF_MAGIC[1]);
    gw.ലിഖ(GGUF_MAGIC[2]);
    gw.ലിഖ(GGUF_MAGIC[3]);
    gw.ലിഖ(ctx->version);
    gw.ലിഖ(n_tensors);
    gw.ലിഖ(n_kv);

    // write key-value pairs
    വേണ്ടി (int64_t i = 0; i < n_kv; ++i) {
        gw.ലിഖ(ctx->kv[i]);
    }

    // write tensor info
    വേണ്ടി (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    എങ്കിൽ (only_meta) {
        തിരികെ;
    }

    സ്ഥിരം size_t offset_data = gw.written_bytes;

    // write tensor data
    വേണ്ടി (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

ശൂന്യം gguf_write_to_buf(സ്ഥിരം ഘടന gguf_context * ctx, std::vector<int8_t> & buf, തർക്കം only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

തർക്കം gguf_write_to_file_ptr(സ്ഥിരം ഘടന gguf_context * ctx, FILE * file, തർക്കം only_meta) {
    GGML_ASSERT(file);

    ശ്രമിക്കുക {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } പിടിക്കുക (സ്ഥിരം std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

തർക്കം gguf_write_to_file(സ്ഥിരം ഘടന gguf_context * ctx, സ്ഥിരം അക്ഷരം * fname, തർക്കം only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    എങ്കിൽ (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        തിരികെ അസത്യം;
    }

    സ്ഥിരം തർക്കം success = gguf_write_to_file_ptr(ctx, file, only_meta);
    എങ്കിൽ (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    തിരികെ success;
}

size_t gguf_get_meta_size(സ്ഥിരം ഘടന gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ സത്യം);
    തിരികെ buf.size();
}

ശൂന്യം gguf_get_meta_data(സ്ഥിരം ഘടന gguf_context * ctx, ശൂന്യം * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ സത്യം);
    memcpy(data, buf.data(), buf.size());
}
