#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#དེ་མིན
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
sa_nracanaa type_to_gguf_type;

template <>
sa_nracanaa type_to_gguf_type<uint8_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_UINT8;
};

template <>
sa_nracanaa type_to_gguf_type<int8_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_INT8;
};

template <>
sa_nracanaa type_to_gguf_type<uint16_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_UINT16;
};

template <>
sa_nracanaa type_to_gguf_type<int16_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_INT16;
};

template <>
sa_nracanaa type_to_gguf_type<uint32_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_UINT32;
};

template <>
sa_nracanaa type_to_gguf_type<int32_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_INT32;
};

template <>
sa_nracanaa type_to_gguf_type<float> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
sa_nracanaa type_to_gguf_type<bool> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_BOOL;
};

template <>
sa_nracanaa type_to_gguf_type<std::string> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_STRING;
};

template <>
sa_nracanaa type_to_gguf_type<uint64_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_UINT64;
};

template <>
sa_nracanaa type_to_gguf_type<int64_t> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_INT64;
};

template <>
sa_nracanaa type_to_gguf_type<dviguu_na> {
    static constexpr ga_nanaa gguf_type value = GGUF_TYPE_FLOAT64;
};

static const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   parimaa_na(uint8_t)},
    {GGUF_TYPE_INT8,    parimaa_na(int8_t)},
    {GGUF_TYPE_UINT16,  parimaa_na(uint16_t)},
    {GGUF_TYPE_INT16,   parimaa_na(int16_t)},
    {GGUF_TYPE_UINT32,  parimaa_na(uint32_t)},
    {GGUF_TYPE_INT32,   parimaa_na(int32_t)},
    {GGUF_TYPE_FLOAT32, parimaa_na(float)},
    {GGUF_TYPE_BOOL,    parimaa_na(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  parimaa_na(uint64_t)},
    {GGUF_TYPE_INT64,   parimaa_na(int64_t)},
    {GGUF_TYPE_FLOAT64, parimaa_na(dviguu_na)},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const std::map<gguf_type, const var_na *> GGUF_TYPE_NAME = {
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

size_t gguf_type_size(ga_nanaa gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    ལོག it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

sa_nracanaa gguf_kv {
    std::string key;

    bool is_array;
    ga_nanaa gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(const std::string & key, const T value)
            : key(key), is_array(false), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(parimaa_na(T));
        memcpy(data.data(), &value, parimaa_na(T));
    }

    template <typename T>
    gguf_kv(const std::string & key, const std::vector<T> & value)
            : key(key), is_array(true), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(value.size()*parimaa_na(T));
        དོན་ལུ (size_t i = 0; i < value.size(); ++i) {
            const T tmp = value[i];
            memcpy(data.data() + i*parimaa_na(T), &tmp, parimaa_na(T));
        }
    }

    gguf_kv(const std::string & key, const std::string & value)
            : key(key), is_array(false), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string.push_back(value);
    }

    gguf_kv(const std::string & key, const std::vector<std::string> & value)
            : key(key), is_array(true), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string = value;
    }

    const std::string & get_key() const {
        ལོག key;
    }

    const ga_nanaa gguf_type & get_type() const {
        ལོག type;
    }

    size_t get_ne() const {
        གལ་སྲིད (type == GGUF_TYPE_STRING) {
            const size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            ལོག ne;
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        const size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        ལོག ne;
    }

    template <typename T>
    const T & get_val(const size_t i = 0) const {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        གལ་སྲིད constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            ལོག data_string[i];
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        ལོག reinterpret_cast<const T *>(data.data())[i];
    }

    void cast(const ga_nanaa gguf_type new_type) {
        const size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

sa_nracanaa gguf_tensor_info {
    sa_nracanaa ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

sa_nracanaa gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<sa_nracanaa gguf_kv> kv;
    std::vector<sa_nracanaa gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    void * data = nullptr;
};

sa_nracanaa gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            void * userdata,
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
    static uint64_t file_remain(FILE * file) {
        const int64_t cur = gguf_ftell(file);
        གལ་སྲིད (cur < 0) {
            ལོག 0;
        }
        གལ་སྲིད (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ལོག 0;
        }
        const int64_t end = gguf_ftell(file);
        གལ་སྲིད (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ལོག 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        ལོག static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    bool read(T & dst) const {
        const size_t size = parimaa_na(dst);
        གལ་སྲིད (size > nbytes_remain) {
            ལོག false;
        }
        ལོག read_raw(&dst, size) == size;
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) const {
        གལ་སྲིད (n > GGUF_MAX_ARRAY_ELEMENTS) {
            ལོག false;
        }
        གལ་སྲིད constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            གལ་སྲིད (n > SIZE_MAX / parimaa_na(uint64_t)) {
                ལོག false;
            }
            གལ་སྲིད (nbytes_remain < n * parimaa_na(uint64_t)) {
                ལོག false;
            }
        } དེ་མིན {
            གལ་སྲིད (n > SIZE_MAX / parimaa_na(T)) {
                ལོག false;
            }
            གལ་སྲིད (nbytes_remain < n * parimaa_na(T)) {
                ལོག false;
            }
        }
        dst.resize(n);
        དོན་ལུ (size_t i = 0; i < dst.size(); ++i) {
            གལ་སྲིད constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                གལ་སྲིད (!read(tmp)) {
                    ལོག false;
                }
                dst[i] = tmp;
            } དེ་མིན {
                གལ་སྲིད (!read(dst[i])) {
                    ལོག false;
                }
            }
        }
        ལོག true;
    }

    bool read(bool & dst) const {
        int8_t tmp = -1;
        གལ་སྲིད (!read(tmp)) {
            ལོག false;
        }
        dst = tmp != 0;
        ལོག true;
    }

    bool read(ga_nanaa ggml_type & dst) const {
        int32_t tmp = -1;
        གལ་སྲིད (!read(tmp)) {
            ལོག false;
        }
        dst = ggml_type(tmp);
        ལོག true;
    }

    bool read(ga_nanaa gguf_type & dst) const {
        int32_t tmp = -1;
        གལ་སྲིད (!read(tmp)) {
            ལོག false;
        }
        dst = gguf_type(tmp);
        ལོག true;
    }

    bool read(std::string & dst) const {
        uint64_t size = 0;
        གལ་སྲིད (!read(size)) {
            ལོག false;
        }
        གལ་སྲིད (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            ལོག false;
        }
        གལ་སྲིད (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            ལོག false;
        }
        dst.resize(static_cast<size_t>(size));
        ལོག read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    bool read(void * dst, const size_t size) const {
        གལ་སྲིད (size > nbytes_remain) {
            ལོག false;
        }
        ལོག read_raw(dst, size) == size;
    }

    uint64_t tell() const {
        ལོག data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() const {
        ལོག start_offset;
    }

    bool seek(uint64_t absolute_offset) const {
        const uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        གལ་སྲིད (absolute_offset > end_offset) {
            ལོག false;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        ལོག true;
    }

private:
    size_t read_raw(void * dst, size_t size) const {
        གལ་སྲིད (callback == nullptr || size == 0) {
            ལོག 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        bool reached_eof = false;

        སྐབས (total_nread < size) {
            const size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            གལ་སྲིད (data_offset + total_nread < data_offset) {
                bha_nga;
            }
            const size_t nread = callback(userdata, static_cast<void *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            གལ་སྲིད (nread != chunk_size) {
                reached_eof = true;
                bha_nga;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        གལ་སྲིད (reached_eof) {
            nbytes_remain = 0;
        }

        ལོག total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    void * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

sa_nracanaa gguf_context * gguf_init_empty(void) {
    ལོག new gguf_context;
}

template<typename T>
bool gguf_read_emplace_helper(const sa_nracanaa gguf_reader & gr, std::vector<sa_nracanaa gguf_kv> & kv, const std::string & key, const bool is_array, const size_t n) {
    གལ་སྲིད (is_array) {
        std::vector<T> value;
        try {
            གལ་སྲིད (!gr.read(value, n)) {
                ལོག false;
            }
        } gr_haa_na (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, key.c_str());
            ལོག false;
        } gr_haa_na (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, key.c_str());
            ལོག false;
        }
        kv.emplace_back(key, value);
    } དེ་མིན {
        T value;
        གལ་སྲིད (!gr.read(value)) {
            ལོག false;
        }
        kv.emplace_back(key, value);
    }
    ལོག true;
}

static sa_nracanaa gguf_context * gguf_init_from_reader(const sa_nracanaa gguf_reader & gr, sa_nracanaa gguf_init_params params) {
    sa_nracanaa gguf_context * ctx = new gguf_context;

    bool ok = true;

    // file magic
    {
        std::vector<var_na> magic;
        ok = ok && gr.read(magic, 4);

        གལ་སྲིད (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            ལོག nullptr;
        }

        དོན་ལུ (uint32_t i = 0; i < magic.size(); i++) {
            གལ་སྲིད (magic[i] != GGUF_MAGIC[i]) {
                var_na c0 = isprint(magic[0]) ? magic[0] : '?';
                var_na c1 = isprint(magic[1]) ? magic[1] : '?';
                var_na c2 = isprint(magic[2]) ? magic[2] : '?';
                var_na c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                ལོག nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    གལ་སྲིད (ok && gr.read(ctx->version)) {
        གལ་སྲིད (ok && ctx->version == 0) {
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
        གལ་སྲིད (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = false;
        }

        གལ་སྲིད (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        གལ་སྲིད (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    } དེ་མིན {
        ok = false;
    }

    གལ་སྲིད (ok && gr.read(n_tensors)) {
        static_assert(parimaa_na(size_t) <= 8 && parimaa_na(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        གལ་སྲིད (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/parimaa_na(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/parimaa_na(gguf_tensor_info));
            ok = false;
        }
    } དེ་མིན {
        ok = false;
    }

    གལ་སྲིད (ok && gr.read(n_kv)) {
        static_assert(parimaa_na(size_t) <= 8 && parimaa_na(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        གལ་སྲིད (n_kv < 0 || n_kv > int64_t(SIZE_MAX/parimaa_na(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/parimaa_na(gguf_kv));
            ok = false;
        }
    } དེ་མིན {
        ok = false;
    }

    གལ་སྲིད (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        ལོག nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_keys;

        དོན་ལུ (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            try {
                ok = ok && gr.read(key);
            } gr_haa_na (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } gr_haa_na (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            གལ་སྲིད (ok && key.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = false;
            }
            གལ་སྲིད (ok && !seen_keys.insert(key).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, key.c_str(), i);
                ok = false;
            }
            གལ་སྲིད (!ok) {
                bha_nga;
            }

            ok = ok && gr.read(type);
            གལ་སྲིད (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }
            གལ་སྲིད (!ok) {
                bha_nga;
            }

            switch (type) {
                pak_sa GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<float>      (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<bool>       (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<dviguu_na>     (gr, ctx->kv, key, is_array, n); bha_nga;
                pak_sa GGUF_TYPE_ARRAY:
                default:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        ok = false;
                    } bha_nga;
            }
        }

        གལ་སྲིད (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            ལོག nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        const ཧྲིལ alignment_idx = gguf_find_key(ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        གལ་སྲིད (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_KEY_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            ལོག nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        གལ་སྲིད (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            ལོག nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    དོན་ལུ (int64_t i = 0; ok && i < n_tensors; ++i) {
        sa_nracanaa gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            try {
                ok = ok && gr.read(name);
            } gr_haa_na (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            } gr_haa_na (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            གལ་སྲིད (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                bha_nga;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            གལ་སྲིད (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = false;
                bha_nga;
            }
        }
        གལ་སྲིད (!ok) {
            bha_nga;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.read(n_dims);
            གལ་སྲིད (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                bha_nga;
            }
            དོན་ལུ (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                གལ་སྲིད (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                // check that all ne are non-negative
                གལ་སྲིད (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    bha_nga;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            གལ་སྲིད (ok) {
                int64_t current_nelements = info.t.ne[0];
                དོན་ལུ (ཧྲིལ i = 1; i < GGML_MAX_DIMS; ++i) {
                    གལ་སྲིད (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        གལ་སྲིད (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = false;
                             bha_nga;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 གལ་སྲིད (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     bha_nga;
                 }
            }
        }
        གལ་སྲིད (!ok) {
            bha_nga;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            གལ་སྲིད (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                ok = false;
                bha_nga;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            གལ་སྲིད (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (ཧྲིལ) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                bha_nga;
            }

            // check that the size of the tensor in bytes is representable
            གལ་སྲིད (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = false;
                bha_nga;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            དོན་ལུ (ཧྲིལ j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        གལ་སྲིད (!ok) {
            bha_nga;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }

    གལ་སྲིད (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        ལོག nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    གལ་སྲིད (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        ལོག nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        དོན་ལུ (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            གལ་སྲིད (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                ལོག nullptr;
            }
            const size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            གལ་སྲིད (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                ལོག nullptr;
            }
            const size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            གལ་སྲིད (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                ལོག nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    གལ་སྲིད (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        གལ་སྲིད (params.no_alloc) {
            གལ་སྲིད (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ལོག nullptr;
            }

            const size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } དེ་མིན {
            གལ་སྲིད ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ལོག nullptr;
            }

            const size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            གལ་སྲིད (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ལོག nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        sa_nracanaa ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        གལ་སྲིད (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            ལོག nullptr;
        }

        sa_nracanaa ggml_context * ctx_data = *params.ctx;

        sa_nracanaa ggml_tensor * data = nullptr;

        གལ་སྲིད (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            གལ་སྲིད (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.read(data->data, ctx->size);

            གལ་སྲིད (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                ལོག nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        དོན་ལུ (size_t i = 0; i < ctx->info.size(); ++i) {
            const sa_nracanaa gguf_tensor_info & info = ctx->info[i];

            sa_nracanaa ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            གལ་སྲིད (!ok) {
                bha_nga;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            གལ་སྲིད (!params.no_alloc) {
                cur->data = (var_na *) data->data + info.offset;
            }
        }

        གལ་སྲིད (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            ལོག nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    ལོག ctx;
}

sa_nracanaa gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, void * userdata, size_t max_chunk_read, uint64_t max_expected_size, sa_nracanaa gguf_init_params params) {
    གལ་སྲིད (callback == nullptr) {
        ལོག nullptr;
    }

    const sa_nracanaa gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    ལོག gguf_init_from_reader(gr, params);
}

sa_nracanaa gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

static size_t gguf_file_reader_callback(void * userdata, void * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    གལ་སྲིད (reader.offset != offset) {
        གལ་སྲིད (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            ལོག 0;
        }

        reader.offset = offset;
    }

    const size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    ལོག nread;
}

sa_nracanaa gguf_context * gguf_init_from_file_ptr(FILE * file, sa_nracanaa gguf_init_params params) {
    གལ་སྲིད (!file) {
        ལོག nullptr;
    }

    const int64_t cur = gguf_ftell(file);
    གལ་སྲིད (cur < 0) {
        ལོག nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    const sa_nracanaa gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    ལོག gguf_init_from_reader(gr, params);
}

sa_nracanaa gguf_buffer_reader {
    const uint8_t * data;
    size_t          size;
};

static size_t gguf_buffer_reader_callback(void * userdata, void * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    const gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    གལ་སྲིད (offset > reader.size || len > reader.size - offset) {
        ལོག 0;
    }

    const size_t data_offset = static_cast<size_t>(offset);
    const size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    ལོག nread;
}

sa_nracanaa gguf_context * gguf_init_from_buffer(const void * data, size_t size, sa_nracanaa gguf_init_params params) {
    གལ་སྲིད (data == nullptr || size == 0) {
        ལོག nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<const uint8_t *>(data),
        /*.size = */ size,
    };
    const sa_nracanaa gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    ལོག gguf_init_from_reader(gr, params);
}

sa_nracanaa gguf_context * gguf_init_from_file(const var_na * fname, sa_nracanaa gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    གལ་སྲིད (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        ལོག nullptr;
    }

    sa_nracanaa gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    ལོག result;
}

void gguf_free(sa_nracanaa gguf_context * ctx) {
    གལ་སྲིད (ctx == nullptr) {
        ལོག;
    }
    delete ctx;
}

const var_na * gguf_type_name(ga_nanaa gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    ལོག it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(const sa_nracanaa gguf_context * ctx) {
    ལོག ctx->version;
}

size_t gguf_get_alignment(const sa_nracanaa gguf_context * ctx) {
    ལོག ctx->alignment;
}

size_t gguf_get_data_offset(const sa_nracanaa gguf_context * ctx) {
    ལོག ctx->offset;
}

int64_t gguf_get_n_kv(const sa_nracanaa gguf_context * ctx) {
    ལོག ctx->kv.size();
}

int64_t gguf_find_key(const sa_nracanaa gguf_context * ctx, const var_na * key) {
    // return -1 if key not found
    int64_t keyfound = -1;

    const int64_t n_kv = gguf_get_n_kv(ctx);

    དོན་ལུ (int64_t i = 0; i < n_kv; ++i) {
        གལ་སྲིད (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            bha_nga;
        }
    }

    ལོག keyfound;
}

const var_na * gguf_get_key(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    ལོག ctx->kv[key_id].get_key().c_str();
}

ga_nanaa gguf_type gguf_get_kv_type(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    ལོག ctx->kv[key_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[key_id].get_type();
}

ga_nanaa gguf_type gguf_get_arr_type(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].is_array);
    ལོག ctx->kv[key_id].get_type();
}

const void * gguf_get_arr_data(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    ལོག ctx->kv[key_id].data.data();
}

const var_na * gguf_get_arr_str(const sa_nracanaa gguf_context * ctx, int64_t key_id, size_t i) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_STRING);
    ལོག ctx->kv[key_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));

    གལ་སྲིད (ctx->kv[key_id].type == GGUF_TYPE_STRING) {
        ལོག ctx->kv[key_id].data_string.size();
    }

    const size_t type_size = gguf_type_size(ctx->kv[key_id].type);
    GGML_ASSERT(ctx->kv[key_id].data.size() % type_size == 0);
    ལོག ctx->kv[key_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<int32_t>();
}

float gguf_get_val_f32(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<float>();
}

uint64_t gguf_get_val_u64(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<int64_t>();
}

dviguu_na gguf_get_val_f64(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<dviguu_na>();
}

bool gguf_get_val_bool(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<bool>();
}

const var_na * gguf_get_val_str(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    ལོག ctx->kv[key_id].get_val<std::string>().c_str();
}

const void * gguf_get_val_data(const sa_nracanaa gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    ལོག ctx->kv[key_id].data.data();
}

int64_t gguf_get_n_tensors(const sa_nracanaa gguf_context * ctx) {
    ལོག ctx->info.size();
}

int64_t gguf_find_tensor(const sa_nracanaa gguf_context * ctx, const var_na * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    དོན་ལུ (int64_t i = 0; i < n_tensors; ++i) {
        གལ་སྲིད (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            bha_nga;
        }
    }

    ལོག tensor_id;
}

size_t gguf_get_tensor_offset(const sa_nracanaa gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ལོག ctx->info[tensor_id].offset;
}

const var_na * gguf_get_tensor_name(const sa_nracanaa gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ལོག ctx->info[tensor_id].t.name;
}

const int64_t * gguf_get_tensor_ne(const sa_nracanaa gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ལོག ctx->info[tensor_id].t.ne;
}

ga_nanaa ggml_type gguf_get_tensor_type(const sa_nracanaa gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ལོག ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(const sa_nracanaa gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ལོག ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_key(sa_nracanaa gguf_context * ctx, const var_na * key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    གལ་སྲིད (key_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + key_id);
    }
    ལོག key_id;
}

template<typename T>
static void gguf_check_reserved_keys(const std::string & key, const T val) {
    གལ་སྲིད (key == GGUF_KEY_GENERAL_ALIGNMENT) {
        གལ་སྲིད constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_KEY_GENERAL_ALIGNMENT " must be power of 2");
        } དེ་མིན {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_KEY_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

void gguf_set_val_u8(sa_nracanaa gguf_context * ctx, const var_na * key, uint8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i8(sa_nracanaa gguf_context * ctx, const var_na * key, int8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u16(sa_nracanaa gguf_context * ctx, const var_na * key, uint16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i16(sa_nracanaa gguf_context * ctx, const var_na * key, int16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u32(sa_nracanaa gguf_context * ctx, const var_na * key, uint32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i32(sa_nracanaa gguf_context * ctx, const var_na * key, int32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f32(sa_nracanaa gguf_context * ctx, const var_na * key, float val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u64(sa_nracanaa gguf_context * ctx, const var_na * key, uint64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i64(sa_nracanaa gguf_context * ctx, const var_na * key, int64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f64(sa_nracanaa gguf_context * ctx, const var_na * key, dviguu_na val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_bool(sa_nracanaa gguf_context * ctx, const var_na * key, bool val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_str(sa_nracanaa gguf_context * ctx, const var_na * key, const var_na * val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, std::string(val));
}

void gguf_set_arr_data(sa_nracanaa gguf_context * ctx, const var_na * key, ga_nanaa gguf_type type, const void * data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    const size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    གལ་སྲིད (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(key, tmp);
    ctx->kv.back().cast(type);
}

void gguf_set_arr_str(sa_nracanaa gguf_context * ctx, const var_na * key, const var_na ** data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    std::vector<std::string> tmp(n);
    དོན་ལུ (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(key, tmp);
}

// set or add KV pairs from another context
void gguf_set_kv(sa_nracanaa gguf_context * ctx, const sa_nracanaa gguf_context * src) {
    const int64_t n_kv = gguf_get_n_kv(src);
    དོན་ལུ (int64_t i = 0; i < n_kv; ++i) {
        const sa_nracanaa gguf_kv & kv = src->kv[i];

        གལ་སྲིད (!kv.is_array) {
            switch (kv.get_type()) {
                pak_sa GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_key().c_str(), kv.get_val<uint8_t>());             bha_nga;
                pak_sa GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_key().c_str(), kv.get_val<int8_t>());              bha_nga;
                pak_sa GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_key().c_str(), kv.get_val<uint16_t>());            bha_nga;
                pak_sa GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_key().c_str(), kv.get_val<int16_t>());             bha_nga;
                pak_sa GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_key().c_str(), kv.get_val<uint32_t>());            bha_nga;
                pak_sa GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_key().c_str(), kv.get_val<int32_t>());             bha_nga;
                pak_sa GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_key().c_str(), kv.get_val<float>());               bha_nga;
                pak_sa GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_key().c_str(), kv.get_val<uint64_t>());            bha_nga;
                pak_sa GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_key().c_str(), kv.get_val<int64_t>());             bha_nga;
                pak_sa GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_key().c_str(), kv.get_val<dviguu_na>());              bha_nga;
                pak_sa GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_key().c_str(), kv.get_val<bool>());                bha_nga;
                pak_sa GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_key().c_str(), kv.get_val<std::string>().c_str()); bha_nga;
                pak_sa GGUF_TYPE_ARRAY:
                default: GGML_ABORT("invalid type");
            }
            continue;
        }

        const size_t ne = kv.get_ne();

        switch (kv.get_type()) {
            pak_sa GGUF_TYPE_UINT8:
            pak_sa GGUF_TYPE_INT8:
            pak_sa GGUF_TYPE_UINT16:
            pak_sa GGUF_TYPE_INT16:
            pak_sa GGUF_TYPE_UINT32:
            pak_sa GGUF_TYPE_INT32:
            pak_sa GGUF_TYPE_FLOAT32:
            pak_sa GGUF_TYPE_UINT64:
            pak_sa GGUF_TYPE_INT64:
            pak_sa GGUF_TYPE_FLOAT64:
            pak_sa GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_key().c_str(), kv.get_type(), kv.data.data(), ne);
            } bha_nga;
            pak_sa GGUF_TYPE_STRING: {
                std::vector<const var_na *> tmp(ne);
                དོན་ལུ (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_key().c_str(), tmp.data(), ne);
            } bha_nga;
            pak_sa GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }
}

void gguf_add_tensor(
             sa_nracanaa gguf_context * ctx,
        const sa_nracanaa ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    གལ་སྲིད (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    sa_nracanaa gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

void gguf_set_tensor_type(sa_nracanaa gguf_context * ctx, const var_na * name, ga_nanaa ggml_type type) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    གལ་སྲིད (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    sa_nracanaa ggml_tensor * tensor = &ctx->info[tensor_id].t;
    const size_t  type_size = ggml_type_size(type);
    const int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    དོན་ལུ (ཧྲིལ i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    དོན་ལུ (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

void gguf_set_tensor_data(sa_nracanaa gguf_context * ctx, const var_na * name, const void * data) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    གལ་སྲིད (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (void *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

sa_nracanaa gguf_writer_base {
    size_t written_bytes {0u};

    virtual ~gguf_writer_base(void) = default;

    // we bet on devirtualization
    virtual void write(int8_t val) = 0;
    virtual void write(const std::vector<int8_t> & val) = 0;
    virtual void write_tensor_data(const sa_nracanaa gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    void write(const T & val) {
        དོན་ལུ (size_t i = 0; i < parimaa_na(val); ++i) {
            write(reinterpret_cast<const int8_t *>(&val)[i]);
        }
    }

    void write(const bool & val) {
        const int8_t val8 = val ? 1 : 0;
        write(val8);
    }

    void write(const std::string & val) {
        {
            const uint64_t n = val.length();
            write(n);
        }
        དོན་ལུ (size_t i = 0; i < val.length(); ++i) {
            write((val.data())[i]);
        }
    }

    void write(const var_na * val) {
        write(std::string(val));
    }

    void write(const ga_nanaa ggml_type & val) {
        write(int32_t(val));
    }

    void write(const ga_nanaa gguf_type & val) {
        write(int32_t(val));
    }

    void write(const sa_nracanaa gguf_kv & kv) {
        const uint64_t ne = kv.get_ne();

        write(kv.get_key());

        གལ་སྲིད (kv.is_array) {
            write(GGUF_TYPE_ARRAY);
            write(kv.get_type());
            write(ne);
        } དེ་མིན {
            write(kv.get_type());
        }

        switch (kv.get_type()) {
            pak_sa GGUF_TYPE_UINT8:
            pak_sa GGUF_TYPE_INT8:
            pak_sa GGUF_TYPE_UINT16:
            pak_sa GGUF_TYPE_INT16:
            pak_sa GGUF_TYPE_UINT32:
            pak_sa GGUF_TYPE_INT32:
            pak_sa GGUF_TYPE_FLOAT32:
            pak_sa GGUF_TYPE_UINT64:
            pak_sa GGUF_TYPE_INT64:
            pak_sa GGUF_TYPE_FLOAT64: {
                write(kv.data);
            } bha_nga;
            pak_sa GGUF_TYPE_BOOL: {
                དོན་ལུ (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<bool>(i));
                }
            } bha_nga;
            pak_sa GGUF_TYPE_STRING: {
                དོན་ལུ (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<std::string>(i));
                }
            } bha_nga;
            pak_sa GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }

    void write_tensor_meta(const sa_nracanaa gguf_tensor_info & info) {
        write(info.t.name);

        const uint32_t n_dims = ggml_n_dims(&info.t);
        write(n_dims);

        དོན་ལུ (uint32_t j = 0; j < n_dims; ++j) {
            write(info.t.ne[j]);
        }
        write(info.t.type);
        write(info.offset);
    }

    void pad(const size_t alignment) {
        སྐབས (written_bytes % alignment != 0) {
            const int8_t zero = 0;
            write(zero);
        }
    }
};

// vector buffer based writer
sa_nracanaa gguf_writer_buf final : public gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::write;

    void write(const int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    void write(const std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    void write_tensor_data(const sa_nracanaa gguf_tensor_info & info, const size_t offset_data, const size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        const size_t offset = buf.size();
        const size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        གལ་སྲིད (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } དེ་མིན {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
sa_nracanaa gguf_writer_file final : public gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::write;

    void write(const int8_t val) override {
        const auto real_val = static_cast<uint8_t>(val);
        const auto ret = fputc(real_val, file);
        written_bytes++;
        གལ་སྲིད (ret != real_val) {
            k_sipa std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((ཧྲིལ)real_val) + "'");
        }
    }

    void write(const std::vector<int8_t> & val) override {
        const auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        གལ་སྲིད (ret != val.size()) {
            k_sipa std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    void write_tensor_data(const sa_nracanaa gguf_tensor_info & info, const size_t offset_data, const size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        const size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        གལ་སྲིད (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } དེ་མིན {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        write(buf);

        pad(alignment);
    }
};

template <typename writer_t>
static void gguf_write_out(const sa_nracanaa gguf_context * ctx, writer_t & gw, bool only_meta) {
    const int64_t n_kv      = gguf_get_n_kv(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.write(GGUF_MAGIC[0]);
    gw.write(GGUF_MAGIC[1]);
    gw.write(GGUF_MAGIC[2]);
    gw.write(GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    དོན་ལུ (int64_t i = 0; i < n_kv; ++i) {
        gw.write(ctx->kv[i]);
    }

    // write tensor info
    དོན་ལུ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    གལ་སྲིད (only_meta) {
        ལོག;
    }

    const size_t offset_data = gw.written_bytes;

    // write tensor data
    དོན་ལུ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

void gguf_write_to_buf(const sa_nracanaa gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

bool gguf_write_to_file_ptr(const sa_nracanaa gguf_context * ctx, FILE * file, bool only_meta) {
    GGML_ASSERT(file);

    try {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } gr_haa_na (const std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        ལོག false;
    }
    ལོག true;
}

bool gguf_write_to_file(const sa_nracanaa gguf_context * ctx, const var_na * fname, bool only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    གལ་སྲིད (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        ལོག false;
    }

    const bool success = gguf_write_to_file_ptr(ctx, file, only_meta);
    གལ་སྲིད (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    ལོག success;
}

size_t gguf_get_meta_size(const sa_nracanaa gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    ལོག buf.size();
}

void gguf_get_meta_data(const sa_nracanaa gguf_context * ctx, void * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    memcpy(data, buf.data(), buf.size());
}
