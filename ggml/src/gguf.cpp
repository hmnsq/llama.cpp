#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "gguf.h"

#અન્તર્ભાવ <cinttypes>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <cerrno>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <new>
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <unordered_set>
#અન્તર્ભાવ <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#નહીંતર
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
સંરચના type_to_gguf_type;

template <>
સંરચના type_to_gguf_type<uint8_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_UINT8;
};

template <>
સંરચના type_to_gguf_type<int8_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_INT8;
};

template <>
સંરચના type_to_gguf_type<uint16_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_UINT16;
};

template <>
સંરચના type_to_gguf_type<int16_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_INT16;
};

template <>
સંરચના type_to_gguf_type<uint32_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_UINT32;
};

template <>
સંરચના type_to_gguf_type<int32_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_INT32;
};

template <>
સંરચના type_to_gguf_type<દશાંશ> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
સંરચના type_to_gguf_type<તાર્કિક> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_BOOL;
};

template <>
સંરચના type_to_gguf_type<std::string> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_STRING;
};

template <>
સંરચના type_to_gguf_type<uint64_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_UINT64;
};

template <>
સંરચના type_to_gguf_type<int64_t> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_INT64;
};

template <>
સંરચના type_to_gguf_type<દ્વિગુણ> {
    સ્થિર constexpr ગણના gguf_type value = GGUF_TYPE_FLOAT64;
};

સ્થિર અચલ std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   કદ(uint8_t)},
    {GGUF_TYPE_INT8,    કદ(int8_t)},
    {GGUF_TYPE_UINT16,  કદ(uint16_t)},
    {GGUF_TYPE_INT16,   કદ(int16_t)},
    {GGUF_TYPE_UINT32,  કદ(uint32_t)},
    {GGUF_TYPE_INT32,   કદ(int32_t)},
    {GGUF_TYPE_FLOAT32, કદ(દશાંશ)},
    {GGUF_TYPE_BOOL,    કદ(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  કદ(uint64_t)},
    {GGUF_TYPE_INT64,   કદ(int64_t)},
    {GGUF_TYPE_FLOAT64, કદ(દ્વિગુણ)},
};
static_assert(GGUF_TYPE_ગણતરી == 13, "GGUF_TYPE_COUNT != 13");

સ્થિર અચલ std::map<gguf_type, અચલ અક્ષર *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_ગણતરી == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(ગણના gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    પાછા it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

સંરચના gguf_kv {
    std::string ચાવી;

    તાર્કિક is_array;
    ગણના gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(અચલ std::string & ચાવી, અચલ T value)
            : ચાવી(ચાવી), is_array(ખોટું), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!ચાવી.empty());
        data.resize(કદ(T));
        memcpy(data.data(), &value, કદ(T));
    }

    template <typename T>
    gguf_kv(અચલ std::string & ચાવી, અચલ std::vector<T> & value)
            : ચાવી(ચાવી), is_array(સાચું), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!ચાવી.empty());
        data.resize(value.size()*કદ(T));
        માટે (size_t i = 0; i < value.size(); ++i) {
            અચલ T tmp = value[i];
            memcpy(data.data() + i*કદ(T), &tmp, કદ(T));
        }
    }

    gguf_kv(અચલ std::string & ચાવી, અચલ std::string & value)
            : ચાવી(ચાવી), is_array(ખોટું), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ચાવી.empty());
        data_string.push_back(value);
    }

    gguf_kv(અચલ std::string & ચાવી, અચલ std::vector<std::string> & value)
            : ચાવી(ચાવી), is_array(સાચું), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ચાવી.empty());
        data_string = value;
    }

    અચલ std::string & get_ચાવી() અચલ {
        પાછા ચાવી;
    }

    અચલ ગણના gguf_type & get_type() અચલ {
        પાછા type;
    }

    size_t get_ne() અચલ {
        જો (type == GGUF_TYPE_STRING) {
            અચલ size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            પાછા ne;
        }
        અચલ size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        અચલ size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        પાછા ne;
    }

    template <typename T>
    અચલ T & get_val(અચલ size_t i = 0) અચલ {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        જો constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            પાછા data_string[i];
        }
        અચલ size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        પાછા reinterpret_cast<અચલ T *>(data.data())[i];
    }

    શૂન્ય cast(અચલ ગણના gguf_type new_type) {
        અચલ size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

સંરચના gguf_tensor_info {
    સંરચના ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

સંરચના gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<સંરચના gguf_kv> kv;
    std::vector<સંરચના gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    શૂન્ય * data = nullptr;
};

સંરચના gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            શૂન્ય * userdata,
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
    સ્થિર uint64_t file_remain(FILE * file) {
        અચલ int64_t cur = gguf_ftell(file);
        જો (cur < 0) {
            પાછા 0;
        }
        જો (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            પાછા 0;
        }
        અચલ int64_t end = gguf_ftell(file);
        જો (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            પાછા 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        પાછા static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    તાર્કિક પઠ(T & dst) અચલ {
        અચલ size_t size = કદ(dst);
        જો (size > nbytes_remain) {
            પાછા ખોટું;
        }
        પાછા read_raw(&dst, size) == size;
    }

    template <typename T>
    તાર્કિક પઠ(std::vector<T> & dst, અચલ size_t n) અચલ {
        જો (n > GGUF_MAX_ARRAY_ELEMENTS) {
            પાછા ખોટું;
        }
        જો constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            જો (n > SIZE_MAX / કદ(uint64_t)) {
                પાછા ખોટું;
            }
            જો (nbytes_remain < n * કદ(uint64_t)) {
                પાછા ખોટું;
            }
        } નહીંતર {
            જો (n > SIZE_MAX / કદ(T)) {
                પાછા ખોટું;
            }
            જો (nbytes_remain < n * કદ(T)) {
                પાછા ખોટું;
            }
        }
        dst.resize(n);
        માટે (size_t i = 0; i < dst.size(); ++i) {
            જો constexpr (std::is_same<T, તાર્કિક>::value) {
                તાર્કિક tmp;
                જો (!પઠ(tmp)) {
                    પાછા ખોટું;
                }
                dst[i] = tmp;
            } નહીંતર {
                જો (!પઠ(dst[i])) {
                    પાછા ખોટું;
                }
            }
        }
        પાછા સાચું;
    }

    તાર્કિક પઠ(તાર્કિક & dst) અચલ {
        int8_t tmp = -1;
        જો (!પઠ(tmp)) {
            પાછા ખોટું;
        }
        dst = tmp != 0;
        પાછા સાચું;
    }

    તાર્કિક પઠ(ગણના ggml_type & dst) અચલ {
        int32_t tmp = -1;
        જો (!પઠ(tmp)) {
            પાછા ખોટું;
        }
        dst = ggml_type(tmp);
        પાછા સાચું;
    }

    તાર્કિક પઠ(ગણના gguf_type & dst) અચલ {
        int32_t tmp = -1;
        જો (!પઠ(tmp)) {
            પાછા ખોટું;
        }
        dst = gguf_type(tmp);
        પાછા સાચું;
    }

    તાર્કિક પઠ(std::string & dst) અચલ {
        uint64_t size = 0;
        જો (!પઠ(size)) {
            પાછા ખોટું;
        }
        જો (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            પાછા ખોટું;
        }
        જો (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            પાછા ખોટું;
        }
        dst.resize(static_cast<size_t>(size));
        પાછા read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    તાર્કિક પઠ(શૂન્ય * dst, અચલ size_t size) અચલ {
        જો (size > nbytes_remain) {
            પાછા ખોટું;
        }
        પાછા read_raw(dst, size) == size;
    }

    uint64_t tell() અચલ {
        પાછા data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() અચલ {
        પાછા start_offset;
    }

    તાર્કિક seek(uint64_t absolute_offset) અચલ {
        અચલ uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        જો (absolute_offset > end_offset) {
            પાછા ખોટું;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        પાછા સાચું;
    }

ખાનગી:
    size_t read_raw(શૂન્ય * dst, size_t size) અચલ {
        જો (callback == nullptr || size == 0) {
            પાછા 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        તાર્કિક reached_eof = ખોટું;

        જ્યારે (total_nread < size) {
            અચલ size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            જો (data_offset + total_nread < data_offset) {
                તોડો;
            }
            અચલ size_t nread = callback(userdata, static_cast<શૂન્ય *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            જો (nread != chunk_size) {
                reached_eof = સાચું;
                તોડો;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        જો (reached_eof) {
            nbytes_remain = 0;
        }

        પાછા total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    શૂન્ય * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

સંરચના gguf_context * gguf_init_empty(શૂન્ય) {
    પાછા નવું gguf_context;
}

template<typename T>
તાર્કિક gguf_read_emplace_helper(અચલ સંરચના gguf_reader & gr, std::vector<સંરચના gguf_kv> & kv, અચલ std::string & ચાવી, અચલ તાર્કિક is_array, અચલ size_t n) {
    જો (is_array) {
        std::vector<T> value;
        પ્રયાસ {
            જો (!gr.પઠ(value, n)) {
                પાછા ખોટું;
            }
        } પકડો (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, ચાવી.c_str());
            પાછા ખોટું;
        } પકડો (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, ચાવી.c_str());
            પાછા ખોટું;
        }
        kv.emplace_back(ચાવી, value);
    } નહીંતર {
        T value;
        જો (!gr.પઠ(value)) {
            પાછા ખોટું;
        }
        kv.emplace_back(ચાવી, value);
    }
    પાછા સાચું;
}

સ્થિર સંરચના gguf_context * gguf_init_from_reader(અચલ સંરચના gguf_reader & gr, સંરચના gguf_init_params params) {
    સંરચના gguf_context * ctx = નવું gguf_context;

    તાર્કિક ok = સાચું;

    // file magic
    {
        std::vector<અક્ષર> magic;
        ok = ok && gr.પઠ(magic, 4);

        જો (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            પાછા nullptr;
        }

        માટે (uint32_t i = 0; i < magic.size(); i++) {
            જો (magic[i] != GGUF_MAGIC[i]) {
                અક્ષર c0 = isprint(magic[0]) ? magic[0] : '?';
                અક્ષર c1 = isprint(magic[1]) ? magic[1] : '?';
                અક્ષર c2 = isprint(magic[2]) ? magic[2] : '?';
                અક્ષર c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                પાછા nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    જો (ok && gr.પઠ(ctx->version)) {
        જો (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = ખોટું;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        જો (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = ખોટું;
        }

        જો (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = ખોટું;
        }
        જો (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = ખોટું;
        }
    } નહીંતર {
        ok = ખોટું;
    }

    જો (ok && gr.પઠ(n_tensors)) {
        static_assert(કદ(size_t) <= 8 && કદ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        જો (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/કદ(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/કદ(gguf_tensor_info));
            ok = ખોટું;
        }
    } નહીંતર {
        ok = ખોટું;
    }

    જો (ok && gr.પઠ(n_kv)) {
        static_assert(કદ(size_t) <= 8 && કદ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        જો (n_kv < 0 || n_kv > int64_t(SIZE_MAX/કદ(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/કદ(gguf_kv));
            ok = ખોટું;
        }
    } નહીંતર {
        ok = ખોટું;
    }

    જો (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        પાછા nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_ચાવીઓ;

        માટે (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string ચાવી;
            gguf_type   type     = gguf_type(-1);
            તાર્કિક        is_array = ખોટું;
            uint64_t    n        = 1;

            પ્રયાસ {
                ok = ok && gr.પઠ(ચાવી);
            } પકડો (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = ખોટું;
            } પકડો (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = ખોટું;
            }
            જો (ok && ચાવી.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = ખોટું;
            }
            જો (ok && !seen_ચાવીઓ.insert(ચાવી).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, ચાવી.c_str(), i);
                ok = ખોટું;
            }
            જો (!ok) {
                તોડો;
            }

            ok = ok && gr.પઠ(type);
            જો (type == GGUF_TYPE_ARRAY) {
                is_array = સાચું;
                ok = ok && gr.પઠ(type);
                ok = ok && gr.પઠ(n);
            }
            જો (!ok) {
                તોડો;
            }

            પસંદગી (type) {
                સ્થિતિ GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<દશાંશ>      (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<તાર્કિક>       (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<દ્વિગુણ>     (gr, ctx->kv, ચાવી, is_array, n); તોડો;
                સ્થિતિ GGUF_TYPE_ARRAY:
                મૂળભૂત:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, ચાવી.c_str(), type);
                        ok = ખોટું;
                    } તોડો;
            }
        }

        જો (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            પાછા nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        અચલ પૂર્ણાંક alignment_idx = gguf_find_ચાવી(ctx, GGUF_ચાવી_GENERAL_ALIGNMENT);
        જો (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_ચાવી_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            પાછા nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        જો (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            પાછા nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    માટે (int64_t i = 0; ok && i < n_tensors; ++i) {
        સંરચના gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            પ્રયાસ {
                ok = ok && gr.પઠ(name);
            } પકડો (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ખોટું;
            } પકડો (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ખોટું;
            }
            જો (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = ખોટું;
                તોડો;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            જો (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = ખોટું;
                તોડો;
            }
        }
        જો (!ok) {
            તોડો;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.પઠ(n_dims);
            જો (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = ખોટું;
                તોડો;
            }
            માટે (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                જો (j < n_dims) {
                    ok = ok && gr.પઠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                જો (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = ખોટું;
                    તોડો;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            જો (ok) {
                int64_t current_nelements = info.t.ne[0];
                માટે (પૂર્ણાંક i = 1; i < GGML_MAX_DIMS; ++i) {
                    જો (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        જો (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = ખોટું;
                             તોડો;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 જો (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     તોડો;
                 }
            }
        }
        જો (!ok) {
            તોડો;
        }

        // tensor type
        {
            ok = ok && gr.પઠ(info.t.type);

            // check that tensor type is within defined range
            જો (info.t.type < 0 || info.t.type >= GGML_TYPE_ગણતરી) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_ગણતરી);
                ok = ખોટું;
                તોડો;
            }
            અચલ size_t  type_size = ggml_type_size(info.t.type);
            અચલ int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            જો (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (પૂર્ણાંક) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = ખોટું;
                તોડો;
            }

            // check that the size of the tensor in bytes is representable
            જો (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = ખોટું;
                તોડો;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            માટે (પૂર્ણાંક j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        જો (!ok) {
            તોડો;
        }

        // tensor data offset within buffer
        ok = ok && gr.પઠ(info.offset);

        ctx->info.push_back(info);
    }

    જો (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        પાછા nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    જો (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        પાછા nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        માટે (size_t i = 0; i < ctx->info.size(); ++i) {
            અચલ gguf_tensor_info & ti = ctx->info[i];
            જો (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                પાછા nullptr;
            }
            અચલ size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            જો (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                પાછા nullptr;
            }
            અચલ size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            જો (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                પાછા nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    જો (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        જો (params.no_alloc) {
            જો (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                પાછા nullptr;
            }

            અચલ size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } નહીંતર {
            જો ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                પાછા nullptr;
            }

            અચલ size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            જો (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                પાછા nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        સંરચના ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        જો (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            પાછા nullptr;
        }

        સંરચના ggml_context * ctx_data = *params.ctx;

        સંરચના ggml_tensor * data = nullptr;

        જો (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            જો (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.પઠ(data->data, ctx->size);

            જો (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                પાછા nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, સાચું);

        // create the tensors
        માટે (size_t i = 0; i < ctx->info.size(); ++i) {
            અચલ સંરચના gguf_tensor_info & info = ctx->info[i];

            સંરચના ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            જો (!ok) {
                તોડો;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            જો (!params.no_alloc) {
                cur->data = (અક્ષર *) data->data + info.offset;
            }
        }

        જો (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            પાછા nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    પાછા ctx;
}

સંરચના gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, શૂન્ય * userdata, size_t max_chunk_read, uint64_t max_expected_size, સંરચના gguf_init_params params) {
    જો (callback == nullptr) {
        પાછા nullptr;
    }

    અચલ સંરચના gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    પાછા gguf_init_from_reader(gr, params);
}

સંરચના gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

સ્થિર size_t gguf_file_reader_callback(શૂન્ય * userdata, શૂન્ય * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    જો (reader.offset != offset) {
        જો (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            પાછા 0;
        }

        reader.offset = offset;
    }

    અચલ size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    પાછા nread;
}

સંરચના gguf_context * gguf_init_from_file_ptr(FILE * file, સંરચના gguf_init_params params) {
    જો (!file) {
        પાછા nullptr;
    }

    અચલ int64_t cur = gguf_ftell(file);
    જો (cur < 0) {
        પાછા nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    અચલ સંરચના gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    પાછા gguf_init_from_reader(gr, params);
}

સંરચના gguf_buffer_reader {
    અચલ uint8_t * data;
    size_t          size;
};

સ્થિર size_t gguf_buffer_reader_callback(શૂન્ય * userdata, શૂન્ય * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    અચલ gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    જો (offset > reader.size || len > reader.size - offset) {
        પાછા 0;
    }

    અચલ size_t data_offset = static_cast<size_t>(offset);
    અચલ size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    પાછા nread;
}

સંરચના gguf_context * gguf_init_from_buffer(અચલ શૂન્ય * data, size_t size, સંરચના gguf_init_params params) {
    જો (data == nullptr || size == 0) {
        પાછા nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<અચલ uint8_t *>(data),
        /*.size = */ size,
    };
    અચલ સંરચના gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    પાછા gguf_init_from_reader(gr, params);
}

સંરચના gguf_context * gguf_init_from_file(અચલ અક્ષર * fname, સંરચના gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    જો (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        પાછા nullptr;
    }

    સંરચના gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    પાછા result;
}

શૂન્ય gguf_free(સંરચના gguf_context * ctx) {
    જો (ctx == nullptr) {
        પાછા;
    }
    delete ctx;
}

અચલ અક્ષર * gguf_type_name(ગણના gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    પાછા it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(અચલ સંરચના gguf_context * ctx) {
    પાછા ctx->version;
}

size_t gguf_get_alignment(અચલ સંરચના gguf_context * ctx) {
    પાછા ctx->alignment;
}

size_t gguf_get_data_offset(અચલ સંરચના gguf_context * ctx) {
    પાછા ctx->offset;
}

int64_t gguf_get_n_kv(અચલ સંરચના gguf_context * ctx) {
    પાછા ctx->kv.size();
}

int64_t gguf_find_ચાવી(અચલ સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી) {
    // return -1 if key not found
    int64_t keyfound = -1;

    અચલ int64_t n_kv = gguf_get_n_kv(ctx);

    માટે (int64_t i = 0; i < n_kv; ++i) {
        જો (strcmp(ચાવી, gguf_get_ચાવી(ctx, i)) == 0) {
            keyfound = i;
            તોડો;
        }
    }

    પાછા keyfound;
}

અચલ અક્ષર * gguf_get_ચાવી(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    પાછા ctx->kv[ચાવી_id].get_ચાવી().c_str();
}

ગણના gguf_type gguf_get_kv_type(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    પાછા ctx->kv[ચાવી_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[ચાવી_id].get_type();
}

ગણના gguf_type gguf_get_arr_type(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].is_array);
    પાછા ctx->kv[ચાવી_id].get_type();
}

અચલ શૂન્ય * gguf_get_arr_data(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_type() != GGUF_TYPE_STRING);
    પાછા ctx->kv[ચાવી_id].data.data();
}

અચલ અક્ષર * gguf_get_arr_str(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id, size_t i) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_type() == GGUF_TYPE_STRING);
    પાછા ctx->kv[ચાવી_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));

    જો (ctx->kv[ચાવી_id].type == GGUF_TYPE_STRING) {
        પાછા ctx->kv[ચાવી_id].data_string.size();
    }

    અચલ size_t type_size = gguf_type_size(ctx->kv[ચાવી_id].type);
    GGML_ASSERT(ctx->kv[ચાવી_id].data.size() % type_size == 0);
    પાછા ctx->kv[ચાવી_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<int32_t>();
}

દશાંશ gguf_get_val_f32(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<દશાંશ>();
}

uint64_t gguf_get_val_u64(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<int64_t>();
}

દ્વિગુણ gguf_get_val_f64(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<દ્વિગુણ>();
}

તાર્કિક gguf_get_val_bool(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<તાર્કિક>();
}

અચલ અક્ષર * gguf_get_val_str(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    પાછા ctx->kv[ચાવી_id].get_val<std::string>().c_str();
}

અચલ શૂન્ય * gguf_get_val_data(અચલ સંરચના gguf_context * ctx, int64_t ચાવી_id) {
    GGML_ASSERT(ચાવી_id >= 0 && ચાવી_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ચાવી_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[ચાવી_id].get_type() != GGUF_TYPE_STRING);
    પાછા ctx->kv[ચાવી_id].data.data();
}

int64_t gguf_get_n_tensors(અચલ સંરચના gguf_context * ctx) {
    પાછા ctx->info.size();
}

int64_t gguf_find_tensor(અચલ સંરચના gguf_context * ctx, અચલ અક્ષર * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    અચલ int64_t n_tensors = gguf_get_n_tensors(ctx);

    માટે (int64_t i = 0; i < n_tensors; ++i) {
        જો (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            તોડો;
        }
    }

    પાછા tensor_id;
}

size_t gguf_get_tensor_offset(અચલ સંરચના gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    પાછા ctx->info[tensor_id].offset;
}

અચલ અક્ષર * gguf_get_tensor_name(અચલ સંરચના gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    પાછા ctx->info[tensor_id].t.name;
}

અચલ int64_t * gguf_get_tensor_ne(અચલ સંરચના gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    પાછા ctx->info[tensor_id].t.ne;
}

ગણના ggml_type gguf_get_tensor_type(અચલ સંરચના gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    પાછા ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(અચલ સંરચના gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    પાછા ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_ચાવી(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી) {
    અચલ int64_t ચાવી_id = gguf_find_ચાવી(ctx, ચાવી);
    જો (ચાવી_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + ચાવી_id);
    }
    પાછા ચાવી_id;
}

template<typename T>
સ્થિર શૂન્ય gguf_check_reserved_ચાવીઓ(અચલ std::string & ચાવી, અચલ T val) {
    જો (ચાવી == GGUF_ચાવી_GENERAL_ALIGNMENT) {
        જો constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_ચાવી_GENERAL_ALIGNMENT " must be power of 2");
        } નહીંતર {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_ચાવી_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

શૂન્ય gguf_set_val_u8(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, uint8_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_i8(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, int8_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_u16(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, uint16_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_i16(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, int16_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_u32(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, uint32_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_i32(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, int32_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_f32(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, દશાંશ val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_u64(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, uint64_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_i64(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, int64_t val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_f64(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, દ્વિગુણ val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_bool(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, તાર્કિક val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, val);
}

શૂન્ય gguf_set_val_str(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, અચલ અક્ષર * val) {
    gguf_check_reserved_ચાવીઓ(ચાવી, val);
    gguf_remove_ચાવી(ctx, ચાવી);
    ctx->kv.emplace_back(ચાવી, std::string(val));
}

શૂન્ય gguf_set_arr_data(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, ગણના gguf_type type, અચલ શૂન્ય * data, size_t n) {
    gguf_check_reserved_ચાવીઓ(ચાવી, data);
    gguf_remove_ચાવી(ctx, ચાવી);

    અચલ size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    જો (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(ચાવી, tmp);
    ctx->kv.back().cast(type);
}

શૂન્ય gguf_set_arr_str(સંરચના gguf_context * ctx, અચલ અક્ષર * ચાવી, અચલ અક્ષર ** data, size_t n) {
    gguf_check_reserved_ચાવીઓ(ચાવી, data);
    gguf_remove_ચાવી(ctx, ચાવી);

    std::vector<std::string> tmp(n);
    માટે (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(ચાવી, tmp);
}

// set or add KV pairs from another context
શૂન્ય gguf_set_kv(સંરચના gguf_context * ctx, અચલ સંરચના gguf_context * src) {
    અચલ int64_t n_kv = gguf_get_n_kv(src);
    માટે (int64_t i = 0; i < n_kv; ++i) {
        અચલ સંરચના gguf_kv & kv = src->kv[i];

        જો (!kv.is_array) {
            પસંદગી (kv.get_type()) {
                સ્થિતિ GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_ચાવી().c_str(), kv.get_val<uint8_t>());             તોડો;
                સ્થિતિ GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_ચાવી().c_str(), kv.get_val<int8_t>());              તોડો;
                સ્થિતિ GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_ચાવી().c_str(), kv.get_val<uint16_t>());            તોડો;
                સ્થિતિ GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_ચાવી().c_str(), kv.get_val<int16_t>());             તોડો;
                સ્થિતિ GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_ચાવી().c_str(), kv.get_val<uint32_t>());            તોડો;
                સ્થિતિ GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_ચાવી().c_str(), kv.get_val<int32_t>());             તોડો;
                સ્થિતિ GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_ચાવી().c_str(), kv.get_val<દશાંશ>());               તોડો;
                સ્થિતિ GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_ચાવી().c_str(), kv.get_val<uint64_t>());            તોડો;
                સ્થિતિ GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_ચાવી().c_str(), kv.get_val<int64_t>());             તોડો;
                સ્થિતિ GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_ચાવી().c_str(), kv.get_val<દ્વિગુણ>());              તોડો;
                સ્થિતિ GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_ચાવી().c_str(), kv.get_val<તાર્કિક>());                તોડો;
                સ્થિતિ GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_ચાવી().c_str(), kv.get_val<std::string>().c_str()); તોડો;
                સ્થિતિ GGUF_TYPE_ARRAY:
                મૂળભૂત: GGML_ABORT("invalid type");
            }
            ચાલુ;
        }

        અચલ size_t ne = kv.get_ne();

        પસંદગી (kv.get_type()) {
            સ્થિતિ GGUF_TYPE_UINT8:
            સ્થિતિ GGUF_TYPE_INT8:
            સ્થિતિ GGUF_TYPE_UINT16:
            સ્થિતિ GGUF_TYPE_INT16:
            સ્થિતિ GGUF_TYPE_UINT32:
            સ્થિતિ GGUF_TYPE_INT32:
            સ્થિતિ GGUF_TYPE_FLOAT32:
            સ્થિતિ GGUF_TYPE_UINT64:
            સ્થિતિ GGUF_TYPE_INT64:
            સ્થિતિ GGUF_TYPE_FLOAT64:
            સ્થિતિ GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_ચાવી().c_str(), kv.get_type(), kv.data.data(), ne);
            } તોડો;
            સ્થિતિ GGUF_TYPE_STRING: {
                std::vector<અચલ અક્ષર *> tmp(ne);
                માટે (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_ચાવી().c_str(), tmp.data(), ne);
            } તોડો;
            સ્થિતિ GGUF_TYPE_ARRAY:
            મૂળભૂત: GGML_ABORT("invalid type");
        }
    }
}

શૂન્ય gguf_add_tensor(
             સંરચના gguf_context * ctx,
        અચલ સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    જો (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    સંરચના gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

શૂન્ય gguf_set_tensor_type(સંરચના gguf_context * ctx, અચલ અક્ષર * name, ગણના ggml_type type) {
    અચલ int64_t tensor_id = gguf_find_tensor(ctx, name);
    જો (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    સંરચના ggml_tensor * tensor = &ctx->info[tensor_id].t;
    અચલ size_t  type_size = ggml_type_size(type);
    અચલ int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    માટે (પૂર્ણાંક i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    અચલ int64_t n_tensors = gguf_get_n_tensors(ctx);
    માટે (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

શૂન્ય gguf_set_tensor_data(સંરચના gguf_context * ctx, અચલ અક્ષર * name, અચલ શૂન્ય * data) {
    અચલ int64_t tensor_id = gguf_find_tensor(ctx, name);
    જો (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (શૂન્ય *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

સંરચના gguf_writer_base {
    size_t written_bytes {0u};

    આભાસી ~gguf_writer_base(શૂન્ય) = મૂળભૂત;

    // we bet on devirtualization
    આભાસી શૂન્ય લિખ(int8_t val) = 0;
    આભાસી શૂન્ય લિખ(અચલ std::vector<int8_t> & val) = 0;
    આભાસી શૂન્ય write_tensor_data(અચલ સંરચના gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    શૂન્ય લિખ(અચલ T & val) {
        માટે (size_t i = 0; i < કદ(val); ++i) {
            લિખ(reinterpret_cast<અચલ int8_t *>(&val)[i]);
        }
    }

    શૂન્ય લિખ(અચલ તાર્કિક & val) {
        અચલ int8_t val8 = val ? 1 : 0;
        લિખ(val8);
    }

    શૂન્ય લિખ(અચલ std::string & val) {
        {
            અચલ uint64_t n = val.length();
            લિખ(n);
        }
        માટે (size_t i = 0; i < val.length(); ++i) {
            લિખ((val.data())[i]);
        }
    }

    શૂન્ય લિખ(અચલ અક્ષર * val) {
        લિખ(std::string(val));
    }

    શૂન્ય લિખ(અચલ ગણના ggml_type & val) {
        લિખ(int32_t(val));
    }

    શૂન્ય લિખ(અચલ ગણના gguf_type & val) {
        લિખ(int32_t(val));
    }

    શૂન્ય લિખ(અચલ સંરચના gguf_kv & kv) {
        અચલ uint64_t ne = kv.get_ne();

        લિખ(kv.get_ચાવી());

        જો (kv.is_array) {
            લિખ(GGUF_TYPE_ARRAY);
            લિખ(kv.get_type());
            લિખ(ne);
        } નહીંતર {
            લિખ(kv.get_type());
        }

        પસંદગી (kv.get_type()) {
            સ્થિતિ GGUF_TYPE_UINT8:
            સ્થિતિ GGUF_TYPE_INT8:
            સ્થિતિ GGUF_TYPE_UINT16:
            સ્થિતિ GGUF_TYPE_INT16:
            સ્થિતિ GGUF_TYPE_UINT32:
            સ્થિતિ GGUF_TYPE_INT32:
            સ્થિતિ GGUF_TYPE_FLOAT32:
            સ્થિતિ GGUF_TYPE_UINT64:
            સ્થિતિ GGUF_TYPE_INT64:
            સ્થિતિ GGUF_TYPE_FLOAT64: {
                લિખ(kv.data);
            } તોડો;
            સ્થિતિ GGUF_TYPE_BOOL: {
                માટે (size_t i = 0; i < ne; ++i) {
                    લિખ(kv.get_val<તાર્કિક>(i));
                }
            } તોડો;
            સ્થિતિ GGUF_TYPE_STRING: {
                માટે (size_t i = 0; i < ne; ++i) {
                    લિખ(kv.get_val<std::string>(i));
                }
            } તોડો;
            સ્થિતિ GGUF_TYPE_ARRAY:
            મૂળભૂત: GGML_ABORT("invalid type");
        }
    }

    શૂન્ય write_tensor_meta(અચલ સંરચના gguf_tensor_info & info) {
        લિખ(info.t.name);

        અચલ uint32_t n_dims = ggml_n_dims(&info.t);
        લિખ(n_dims);

        માટે (uint32_t j = 0; j < n_dims; ++j) {
            લિખ(info.t.ne[j]);
        }
        લિખ(info.t.type);
        લિખ(info.offset);
    }

    શૂન્ય pad(અચલ size_t alignment) {
        જ્યારે (written_bytes % alignment != 0) {
            અચલ int8_t zero = 0;
            લિખ(zero);
        }
    }
};

// vector buffer based writer
સંરચના gguf_writer_buf final : જાહેર gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::લિખ;

    શૂન્ય લિખ(અચલ int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    શૂન્ય લિખ(અચલ std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    શૂન્ય write_tensor_data(અચલ સંરચના gguf_tensor_info & info, અચલ size_t offset_data, અચલ size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        અચલ size_t offset = buf.size();
        અચલ size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        જો (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } નહીંતર {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
સંરચના gguf_writer_file final : જાહેર gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::લિખ;

    શૂન્ય લિખ(અચલ int8_t val) override {
        અચલ auto real_val = static_cast<uint8_t>(val);
        અચલ auto ret = fputc(real_val, file);
        written_bytes++;
        જો (ret != real_val) {
            ફેંકો std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((પૂર્ણાંક)real_val) + "'");
        }
    }

    શૂન્ય લિખ(અચલ std::vector<int8_t> & val) override {
        અચલ auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        જો (ret != val.size()) {
            ફેંકો std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    શૂન્ય write_tensor_data(અચલ સંરચના gguf_tensor_info & info, અચલ size_t offset_data, અચલ size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        અચલ size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        જો (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } નહીંતર {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        લિખ(buf);

        pad(alignment);
    }
};

template <typename writer_t>
સ્થિર શૂન્ય gguf_write_out(અચલ સંરચના gguf_context * ctx, writer_t & gw, તાર્કિક only_meta) {
    અચલ int64_t n_kv      = gguf_get_n_kv(ctx);
    અચલ int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.લિખ(GGUF_MAGIC[0]);
    gw.લિખ(GGUF_MAGIC[1]);
    gw.લિખ(GGUF_MAGIC[2]);
    gw.લિખ(GGUF_MAGIC[3]);
    gw.લિખ(ctx->version);
    gw.લિખ(n_tensors);
    gw.લિખ(n_kv);

    // write key-value pairs
    માટે (int64_t i = 0; i < n_kv; ++i) {
        gw.લિખ(ctx->kv[i]);
    }

    // write tensor info
    માટે (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    જો (only_meta) {
        પાછા;
    }

    અચલ size_t offset_data = gw.written_bytes;

    // write tensor data
    માટે (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

શૂન્ય gguf_write_to_buf(અચલ સંરચના gguf_context * ctx, std::vector<int8_t> & buf, તાર્કિક only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

તાર્કિક gguf_write_to_file_ptr(અચલ સંરચના gguf_context * ctx, FILE * file, તાર્કિક only_meta) {
    GGML_ASSERT(file);

    પ્રયાસ {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } પકડો (અચલ std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        પાછા ખોટું;
    }
    પાછા સાચું;
}

તાર્કિક gguf_write_to_file(અચલ સંરચના gguf_context * ctx, અચલ અક્ષર * fname, તાર્કિક only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    જો (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        પાછા ખોટું;
    }

    અચલ તાર્કિક success = gguf_write_to_file_ptr(ctx, file, only_meta);
    જો (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    પાછા success;
}

size_t gguf_get_meta_size(અચલ સંરચના gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ સાચું);
    પાછા buf.size();
}

શૂન્ય gguf_get_meta_data(અચલ સંરચના gguf_context * ctx, શૂન્ય * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ સાચું);
    memcpy(data, buf.data(), buf.size());
}
