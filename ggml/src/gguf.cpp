#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gguf.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <new>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGUF_MAX_STRING_ਲੰਮਾਈ  (1024*1024*1024)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#ਨਹੀਂਤਾਂ
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type;

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<uint8_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_UINT8;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<int8_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_INT8;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<uint16_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_UINT16;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<int16_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_INT16;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<uint32_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_UINT32;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<int32_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_INT32;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<ਦਸ਼ਮਲਵ> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_FLOAT32;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<ਤਾਰਕਿਕ> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_BOOL;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<std::string> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_STRING;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<uint64_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_UINT64;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<int64_t> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_INT64;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_ਮੰਜ਼ਿਲ_gguf_type<ਦੁੱਗਣਾ> {
    ਸਥਿਰ constexpr ਗਿਣਤੀ gguf_type value = GGUF_TYPE_FLOAT64;
};

ਸਥਿਰ ਅਚਲ std::map<gguf_type, size_t> GGUF_TYPE_ਮਾਪ = {
    {GGUF_TYPE_UINT8,   ਆਕਾਰ(uint8_t)},
    {GGUF_TYPE_INT8,    ਆਕਾਰ(int8_t)},
    {GGUF_TYPE_UINT16,  ਆਕਾਰ(uint16_t)},
    {GGUF_TYPE_INT16,   ਆਕਾਰ(int16_t)},
    {GGUF_TYPE_UINT32,  ਆਕਾਰ(uint32_t)},
    {GGUF_TYPE_INT32,   ਆਕਾਰ(int32_t)},
    {GGUF_TYPE_FLOAT32, ਆਕਾਰ(ਦਸ਼ਮਲਵ)},
    {GGUF_TYPE_BOOL,    ਆਕਾਰ(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  ਆਕਾਰ(uint64_t)},
    {GGUF_TYPE_INT64,   ਆਕਾਰ(int64_t)},
    {GGUF_TYPE_FLOAT64, ਆਕਾਰ(ਦੁੱਗਣਾ)},
};
static_assert(GGUF_TYPE_ਗਿਣਤੀ == 13, "GGUF_TYPE_COUNT != 13");

ਸਥਿਰ ਅਚਲ std::map<gguf_type, ਅਚਲ ਅੱਖਰ *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_ਗਿਣਤੀ == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_ਮਾਪ(ਗਿਣਤੀ gguf_type type) {
    ਸਵੈਚਾਲਿਤ it = GGUF_TYPE_ਮਾਪ.find(type);
    ਵਾਪਸ it == GGUF_TYPE_ਮਾਪ.end() ? 0 : it->second;
}

ਬਣਤਰ gguf_kv {
    std::string ਕੁੰਜੀ;

    ਤਾਰਕਿਕ is_array;
    ਗਿਣਤੀ gguf_type type;

    std::vector<int8_t>      ਅੰਕੜੇ;
    std::vector<std::string> ਅੰਕੜੇ_string;

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    gguf_kv(ਅਚਲ std::string & ਕੁੰਜੀ, ਅਚਲ T value)
            : ਕੁੰਜੀ(ਕੁੰਜੀ), is_array(ਝੂਠ), type(type_ਮੰਜ਼ਿਲ_gguf_type<T>::value) {
        GGML_ASSERT(!ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ());
        ਅੰਕੜੇ.resize(ਆਕਾਰ(T));
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ.ਅੰਕੜੇ(), &value, ਆਕਾਰ(T));
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    gguf_kv(ਅਚਲ std::string & ਕੁੰਜੀ, ਅਚਲ std::vector<T> & value)
            : ਕੁੰਜੀ(ਕੁੰਜੀ), is_array(ਸੱਚ), type(type_ਮੰਜ਼ਿਲ_gguf_type<T>::value) {
        GGML_ASSERT(!ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ());
        ਅੰਕੜੇ.resize(value.size()*ਆਕਾਰ(T));
        ਲਈ (size_t i = 0; i < value.size(); ++i) {
            ਅਚਲ T tmp = value[i];
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ.ਅੰਕੜੇ() + i*ਆਕਾਰ(T), &tmp, ਆਕਾਰ(T));
        }
    }

    gguf_kv(ਅਚਲ std::string & ਕੁੰਜੀ, ਅਚਲ std::string & value)
            : ਕੁੰਜੀ(ਕੁੰਜੀ), is_array(ਝੂਠ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ());
        ਅੰਕੜੇ_string.push_back(value);
    }

    gguf_kv(ਅਚਲ std::string & ਕੁੰਜੀ, ਅਚਲ std::vector<std::string> & value)
            : ਕੁੰਜੀ(ਕੁੰਜੀ), is_array(ਸੱਚ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ());
        ਅੰਕੜੇ_string = value;
    }

    ਅਚਲ std::string & get_ਕੁੰਜੀ() ਅਚਲ {
        ਵਾਪਸ ਕੁੰਜੀ;
    }

    ਅਚਲ ਗਿਣਤੀ gguf_type & get_type() ਅਚਲ {
        ਵਾਪਸ type;
    }

    size_t get_ne() ਅਚਲ {
        ਜੇ (type == GGUF_TYPE_STRING) {
            ਅਚਲ size_t ne = ਅੰਕੜੇ_string.size();
            GGML_ASSERT(is_array || ne == 1);
            ਵਾਪਸ ne;
        }
        ਅਚਲ size_t type_ਮਾਪ = gguf_type_ਮਾਪ(type);
        GGML_ASSERT(ਅੰਕੜੇ.size() % type_ਮਾਪ == 0);
        ਅਚਲ size_t ne = ਅੰਕੜੇ.size() / type_ਮਾਪ;
        GGML_ASSERT(is_array || ne == 1);
        ਵਾਪਸ ne;
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਅਚਲ T & get_val(ਅਚਲ size_t i = 0) ਅਚਲ {
        GGML_ASSERT(type_ਮੰਜ਼ਿਲ_gguf_type<T>::value == type);
        ਜੇ constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(ਅੰਕੜੇ_string.size() >= i+1);
            ਵਾਪਸ ਅੰਕੜੇ_string[i];
        }
        ਅਚਲ size_t type_ਮਾਪ = gguf_type_ਮਾਪ(type);
        GGML_ASSERT(ਅੰਕੜੇ.size() % type_ਮਾਪ == 0);
        GGML_ASSERT(ਅੰਕੜੇ.size() >= (i+1)*type_ਮਾਪ);
        ਵਾਪਸ reinterpret_cast<ਅਚਲ T *>(ਅੰਕੜੇ.ਅੰਕੜੇ())[i];
    }

    ਖਾਲੀ cast(ਅਚਲ ਗਿਣਤੀ gguf_type new_type) {
        ਅਚਲ size_t new_type_ਮਾਪ = gguf_type_ਮਾਪ(new_type);
        GGML_ASSERT(ਅੰਕੜੇ.size() % new_type_ਮਾਪ == 0);
        type = new_type;
    }
};

ਬਣਤਰ gguf_tensor_info {
    ਬਣਤਰ ggml_tensor t; // for holding the equivalent info
    uint64_t ਵਿੱਥ;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

ਬਣਤਰ gguf_context {
    uint32_t ਸੰਸਕਰਣ_2 = GGUF_ਸੰਸਕਰਣ;

    std::vector<ਬਣਤਰ gguf_kv> kv;
    std::vector<ਬਣਤਰ gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t ਵਿੱਥ    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    ਖਾਲੀ * ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
};

ਬਣਤਰ gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            ਖਾਲੀ * userdata,
            size_t max_chunk_read,
            uint64_t ਅੰਕੜੇ_ਵਿੱਥ = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          ਸ਼ੁਰੂ_ਵਿੱਥ(ਅੰਕੜੇ_ਵਿੱਥ),
          ਅੰਕੜੇ_ਵਿੱਥ(ਅੰਕੜੇ_ਵਿੱਥ),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    ਸਥਿਰ uint64_t file_remain(FILE * file) {
        ਅਚਲ int64_t cur = gguf_ftell(file);
        ਜੇ (cur < 0) {
            ਵਾਪਸ 0;
        }
        ਜੇ (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ਵਾਪਸ 0;
        }
        ਅਚਲ int64_t end = gguf_ftell(file);
        ਜੇ (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ਵਾਪਸ 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        ਵਾਪਸ static_cast<uint64_t>(end - cur);
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਤਾਰਕਿਕ ਪਠ(T & dst) ਅਚਲ {
        ਅਚਲ size_t size = ਆਕਾਰ(dst);
        ਜੇ (size > nbytes_remain) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ read_raw(&dst, size) == size;
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਤਾਰਕਿਕ ਪਠ(std::vector<T> & dst, ਅਚਲ size_t n) ਅਚਲ {
        ਜੇ (n > GGUF_MAX_ARRAY_ELEMENTS) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            ਜੇ (n > SIZE_MAX / ਆਕਾਰ(uint64_t)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (nbytes_remain < n * ਆਕਾਰ(uint64_t)) {
                ਵਾਪਸ ਝੂਠ;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (n > SIZE_MAX / ਆਕਾਰ(T)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (nbytes_remain < n * ਆਕਾਰ(T)) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        dst.resize(n);
        ਲਈ (size_t i = 0; i < dst.size(); ++i) {
            ਜੇ constexpr (std::is_same<T, ਤਾਰਕਿਕ>::value) {
                ਤਾਰਕਿਕ tmp;
                ਜੇ (!ਪਠ(tmp)) {
                    ਵਾਪਸ ਝੂਠ;
                }
                dst[i] = tmp;
            } ਨਹੀਂਤਾਂ {
                ਜੇ (!ਪਠ(dst[i])) {
                    ਵਾਪਸ ਝੂਠ;
                }
            }
        }
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ ਪਠ(ਤਾਰਕਿਕ & dst) ਅਚਲ {
        int8_t tmp = -1;
        ਜੇ (!ਪਠ(tmp)) {
            ਵਾਪਸ ਝੂਠ;
        }
        dst = tmp != 0;
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ ਪਠ(ਗਿਣਤੀ ggml_type & dst) ਅਚਲ {
        int32_t tmp = -1;
        ਜੇ (!ਪਠ(tmp)) {
            ਵਾਪਸ ਝੂਠ;
        }
        dst = ggml_type(tmp);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ ਪਠ(ਗਿਣਤੀ gguf_type & dst) ਅਚਲ {
        int32_t tmp = -1;
        ਜੇ (!ਪਠ(tmp)) {
            ਵਾਪਸ ਝੂਠ;
        }
        dst = gguf_type(tmp);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ ਪਠ(std::string & dst) ਅਚਲ {
        uint64_t size = 0;
        ਜੇ (!ਪਠ(size)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (size > GGUF_MAX_STRING_ਲੰਮਾਈ) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_ਲੰਮਾਈ);
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            ਵਾਪਸ ਝੂਠ;
        }
        dst.resize(static_cast<size_t>(size));
        ਵਾਪਸ read_raw(dst.ਅੰਕੜੇ(), static_cast<size_t>(size)) == size;
    }

    ਤਾਰਕਿਕ ਪਠ(ਖਾਲੀ * dst, ਅਚਲ size_t size) ਅਚਲ {
        ਜੇ (size > nbytes_remain) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ read_raw(dst, size) == size;
    }

    uint64_t tell() ਅਚਲ {
        ਵਾਪਸ ਅੰਕੜੇ_ਵਿੱਥ;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ਅਚਲ {
        ਵਾਪਸ ਸ਼ੁਰੂ_ਵਿੱਥ;
    }

    ਤਾਰਕਿਕ seek(uint64_t absolute_ਵਿੱਥ) ਅਚਲ {
        ਅਚਲ uint64_t ਸਿਰਾ_ਵਿੱਥ = uint64_t(ਅੰਕੜੇ_ਵਿੱਥ) + nbytes_remain;
        ਜੇ (absolute_ਵਿੱਥ > ਸਿਰਾ_ਵਿੱਥ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅੰਕੜੇ_ਵਿੱਥ = absolute_ਵਿੱਥ;
        nbytes_remain = ਸਿਰਾ_ਵਿੱਥ - absolute_ਵਿੱਥ;

        ਵਾਪਸ ਸੱਚ;
    }

ਨਿਜੀ:
    size_t read_raw(ਖਾਲੀ * dst, size_t size) ਅਚਲ {
        ਜੇ (callback == ਖਾਲੀ_ਸੂਚਕ || size == 0) {
            ਵਾਪਸ 0;
        }

        uint8_t * ਅੰਕੜੇ = static_cast<uint8_t *>(dst);
        size_t ਕੁੱਲ_nread = 0;
        ਤਾਰਕਿਕ reached_eof = ਝੂਠ;

        ਜਦੋਂ (ਕੁੱਲ_nread < size) {
            ਅਚਲ size_t chunk_ਮਾਪ = std::min(max_chunk_read, size - ਕੁੱਲ_nread);
            ਜੇ (ਅੰਕੜੇ_ਵਿੱਥ + ਕੁੱਲ_nread < ਅੰਕੜੇ_ਵਿੱਥ) {
                ਰੋਕੋ;
            }
            ਅਚਲ size_t nread = callback(userdata, static_cast<ਖਾਲੀ *>(ਅੰਕੜੇ + ਕੁੱਲ_nread), ਅੰਕੜੇ_ਵਿੱਥ + ਕੁੱਲ_nread, chunk_ਮਾਪ);
            ਕੁੱਲ_nread += nread;
            ਜੇ (nread != chunk_ਮਾਪ) {
                reached_eof = ਸੱਚ;
                ਰੋਕੋ;
            }
        }

        ਅੰਕੜੇ_ਵਿੱਥ += ਕੁੱਲ_nread;
        GGML_ASSERT(ਕੁੱਲ_nread <= nbytes_remain);
        nbytes_remain -= ਕੁੱਲ_nread;

        ਜੇ (reached_eof) {
            nbytes_remain = 0;
        }

        ਵਾਪਸ ਕੁੱਲ_nread;
    }

    gguf_reader_callback_t callback = ਖਾਲੀ_ਸੂਚਕ;
    ਖਾਲੀ * userdata = ਖਾਲੀ_ਸੂਚਕ;
    size_t max_chunk_read = 0;
    uint64_t ਸ਼ੁਰੂ_ਵਿੱਥ = 0;
    ਬਦਲਣਯੋਗ uint64_t ਅੰਕੜੇ_ਵਿੱਥ = 0;
    ਬਦਲਣਯੋਗ uint64_t nbytes_remain = 0;
};

ਬਣਤਰ gguf_context * gguf_init_ਖਾਲੀ_ਥਾਵਾਂ(ਖਾਲੀ) {
    ਵਾਪਸ ਨਵਾਂ gguf_context;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਤਾਰਕਿਕ gguf_read_emplace_helper(ਅਚਲ ਬਣਤਰ gguf_reader & gr, std::vector<ਬਣਤਰ gguf_kv> & kv, ਅਚਲ std::string & ਕੁੰਜੀ, ਅਚਲ ਤਾਰਕਿਕ is_array, ਅਚਲ size_t n) {
    ਜੇ (is_array) {
        std::vector<T> value;
        ਕੋਸ਼ਿਸ਼ {
            ਜੇ (!gr.ਪਠ(value, n)) {
                ਵਾਪਸ ਝੂਠ;
            }
        } ਫੜੋ (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, ਕੁੰਜੀ.c_str());
            ਵਾਪਸ ਝੂਠ;
        } ਫੜੋ (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, ਕੁੰਜੀ.c_str());
            ਵਾਪਸ ਝੂਠ;
        }
        kv.emplace_back(ਕੁੰਜੀ, value);
    } ਨਹੀਂਤਾਂ {
        T value;
        ਜੇ (!gr.ਪਠ(value)) {
            ਵਾਪਸ ਝੂਠ;
        }
        kv.emplace_back(ਕੁੰਜੀ, value);
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਬਣਤਰ gguf_context * gguf_init_from_reader(ਅਚਲ ਬਣਤਰ gguf_reader & gr, ਬਣਤਰ gguf_init_params params) {
    ਬਣਤਰ gguf_context * ctx = ਨਵਾਂ gguf_context;

    ਤਾਰਕਿਕ ok = ਸੱਚ;

    // file magic
    {
        std::vector<ਅੱਖਰ> magic;
        ok = ok && gr.ਪਠ(magic, 4);

        ਜੇ (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ਲਈ (uint32_t i = 0; i < magic.size(); i++) {
            ਜੇ (magic[i] != GGUF_MAGIC[i]) {
                ਅੱਖਰ c0 = isprint(magic[0]) ? magic[0] : '?';
                ਅੱਖਰ c1 = isprint(magic[1]) ? magic[1] : '?';
                ਅੱਖਰ c2 = isprint(magic[2]) ? magic[2] : '?';
                ਅੱਖਰ c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }
    }

    //ਸਿਰਲੇਖ
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    ਜੇ (ok && gr.ਪਠ(ctx->ਸੰਸਕਰਣ_2)) {
        ਜੇ (ok && ctx->ਸੰਸਕਰਣ_2 == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->ਸੰਸਕਰਣ_2);
            ok = ਝੂਠ;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        ਜੇ (ok && (ctx->ਸੰਸਕਰਣ_2 & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->ਸੰਸਕਰਣ_2);
            ok = ਝੂਠ;
        }

        ਜੇ (ok && ctx->ਸੰਸਕਰਣ_2 == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = ਝੂਠ;
        }
        ਜੇ (ok && ctx->ਸੰਸਕਰਣ_2 > GGUF_ਸੰਸਕਰਣ) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->ਸੰਸਕਰਣ_2, GGUF_ਸੰਸਕਰਣ);
            ok = ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ok = ਝੂਠ;
    }

    ਜੇ (ok && gr.ਪਠ(n_tensors)) {
        static_assert(ਆਕਾਰ(size_t) <= 8 && ਆਕਾਰ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ਜੇ (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/ਆਕਾਰ(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/ਆਕਾਰ(gguf_tensor_info));
            ok = ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ok = ਝੂਠ;
    }

    ਜੇ (ok && gr.ਪਠ(n_kv)) {
        static_assert(ਆਕਾਰ(size_t) <= 8 && ਆਕਾਰ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ਜੇ (n_kv < 0 || n_kv > int64_t(SIZE_MAX/ਆਕਾਰ(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/ਆਕਾਰ(gguf_kv));
            ok = ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ok = ਝੂਠ;
    }

    ਜੇ (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    // KV pairs
    {
        std::unordered_set<std::string> ਵੇਖਿਆ_ਕੁੰਜੀਆਂ;

        ਲਈ (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string ਕੁੰਜੀ;
            gguf_type   type     = gguf_type(-1);
            ਤਾਰਕਿਕ        is_array = ਝੂਠ;
            uint64_t    n        = 1;

            ਕੋਸ਼ਿਸ਼ {
                ok = ok && gr.ਪਠ(ਕੁੰਜੀ);
            } ਫੜੋ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = ਝੂਠ;
            } ਫੜੋ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = ਝੂਠ;
            }
            ਜੇ (ok && ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = ਝੂਠ;
            }
            ਜੇ (ok && !ਵੇਖਿਆ_ਕੁੰਜੀਆਂ.insert(ਕੁੰਜੀ).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, ਕੁੰਜੀ.c_str(), i);
                ok = ਝੂਠ;
            }
            ਜੇ (!ok) {
                ਰੋਕੋ;
            }

            ok = ok && gr.ਪਠ(type);
            ਜੇ (type == GGUF_TYPE_ARRAY) {
                is_array = ਸੱਚ;
                ok = ok && gr.ਪਠ(type);
                ok = ok && gr.ਪਠ(n);
            }
            ਜੇ (!ok) {
                ਰੋਕੋ;
            }

            ਚੋਣ (type) {
                ਹਾਲਤ GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<ਦਸ਼ਮਲਵ>      (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<ਤਾਰਕਿਕ>       (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<ਦੁੱਗਣਾ>     (gr, ctx->kv, ਕੁੰਜੀ, is_array, n); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_ARRAY:
                ਮੂਲ:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, ਕੁੰਜੀ.c_str(), type);
                        ok = ਝੂਠ;
                    } ਰੋਕੋ;
            }
        }

        ਜੇ (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ਅਚਲ ਪੂਰਨ alignment_idx = gguf_ਲੱਭੋ_ਕੁੰਜੀ(ctx, GGUF_ਕੁੰਜੀ_GENERAL_ALIGNMENT);
        ਜੇ (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_ਕੁੰਜੀ_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        ਜੇ (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> ਵੇਖਿਆ_tensor_ਨਾਂ;

    ਲਈ (int64_t i = 0; ok && i < n_tensors; ++i) {
        ਬਣਤਰ gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            ਕੋਸ਼ਿਸ਼ {
                ok = ok && gr.ਪਠ(name);
            } ਫੜੋ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ਝੂਠ;
            } ਫੜੋ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ਝੂਠ;
            }
            ਜੇ (name.ਲੰਮਾਈ() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.ਲੰਮਾਈ(), GGML_MAX_NAME);
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }
            ggml_ਸੈੱਟ_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            ਜੇ (ok && !ਵੇਖਿਆ_tensor_ਨਾਂ.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (!ok) {
            ਰੋਕੋ;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.ਪਠ(n_dims);
            ਜੇ (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }
            ਲਈ (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                ਜੇ (j < n_dims) {
                    ok = ok && gr.ਪਠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                ਜੇ (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = ਝੂਠ;
                    ਰੋਕੋ;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            ਜੇ (ok) {
                int64_t current_nelements = info.t.ne[0];
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
                    ਜੇ (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        ਜੇ (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = ਝੂਠ;
                             ਰੋਕੋ;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 ਜੇ (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     ਰੋਕੋ;
                 }
            }
        }
        ਜੇ (!ok) {
            ਰੋਕੋ;
        }

        // tensor type
        {
            ok = ok && gr.ਪਠ(info.t.type);

            // check that tensor type is within defined range
            ਜੇ (info.t.type < 0 || info.t.type >= GGML_TYPE_ਗਿਣਤੀ) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_ਗਿਣਤੀ);
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }
            ਅਚਲ size_t  type_ਮਾਪ = ggml_type_ਮਾਪ(info.t.type);
            ਅਚਲ int64_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(info.t.type);

            // check that row size is divisible by block size
            ਜੇ (blck_ਮਾਪ == 0 || info.t.ne[0] % blck_ਮਾਪ != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (ਪੂਰਨ) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_ਮਾਪ);
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }

            // check that the size of the tensor in bytes is representable
            ਜੇ (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_ਮਾਪ(info.t.type)) > SIZE_MAX/ggml_type_ਮਾਪ(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_ਮਾਪ;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_ਮਾਪ);
            ਲਈ (ਪੂਰਨ j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        ਜੇ (!ok) {
            ਰੋਕੋ;
        }

        // tensor data offset within buffer
        ok = ok && gr.ਪਠ(info.ਵਿੱਥ);

        ctx->info.push_back(info);
    }

    ਜੇ (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    ਜੇ (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    // store the current file offset - this is where the data section starts
    ctx->ਵਿੱਥ = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        ਲਈ (size_t i = 0; i < ctx->info.size(); ++i) {
            ਅਚਲ gguf_tensor_info & ti = ctx->info[i];
            ਜੇ (ti.ਵਿੱਥ != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.ਵਿੱਥ, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ਅਚਲ size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            ਜੇ (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ਅਚਲ size_t padded_ਮਾਪ = GGML_PAD(nbytes, ctx->alignment);
            ਜੇ (SIZE_MAX - ctx->size < padded_ਮਾਪ) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_ਮਾਪ);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ctx->size += padded_ਮਾਪ;
        }
    }

    // load the tensor data only if requested
    ਜੇ (params.ctx != ਖਾਲੀ_ਸੂਚਕ) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_ਮਾਪ = 0;
        ਜੇ (params.no_alloc) {
            ਜੇ (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }

            ਅਚਲ size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_ਮਾਪ = overhead;
        } ਨਹੀਂਤਾਂ {
            ਜੇ ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }

            ਅਚਲ size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            ਜੇ (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }

            mem_ਮਾਪ = overhead + ctx->size;
        }

        ਬਣਤਰ ggml_init_params pdata = {
            /*mem_size   =*/ mem_ਮਾਪ,
            /*mem_buffer =*/ ਖਾਲੀ_ਸੂਚਕ,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        ਜੇ (*params.ctx == ਖਾਲੀ_ਸੂਚਕ) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ਬਣਤਰ ggml_context * ctx_ਅੰਕੜੇ = *params.ctx;

        ਬਣਤਰ ggml_tensor * ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;

        ਜੇ (!params.no_alloc) {
            ਅੰਕੜੇ = ggml_new_tensor_1d(ctx_ਅੰਕੜੇ, GGML_TYPE_I8, ctx->size);

            ok = ok && ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ;

            ਜੇ (ok) {
                ggml_ਸੈੱਟ_name(ਅੰਕੜੇ, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.ਪਠ(ਅੰਕੜੇ->ਅੰਕੜੇ, ctx->size);

            ਜੇ (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_ਅੰਕੜੇ);
                *params.ctx = ਖਾਲੀ_ਸੂਚਕ;
                gguf_free(ctx);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }

            ctx->ਅੰਕੜੇ = ਅੰਕੜੇ->ਅੰਕੜੇ;
        }

        ggml_ਸੈੱਟ_no_alloc(ctx_ਅੰਕੜੇ, ਸੱਚ);

        // create the tensors
        ਲਈ (size_t i = 0; i < ctx->info.size(); ++i) {
            ਅਚਲ ਬਣਤਰ gguf_tensor_info & info = ctx->info[i];

            ਬਣਤਰ ggml_tensor * cur = ggml_new_tensor(ctx_ਅੰਕੜੇ, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != ਖਾਲੀ_ਸੂਚਕ;

            ਜੇ (!ok) {
                ਰੋਕੋ;
            }

            ggml_ਸੈੱਟ_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            ਜੇ (!params.no_alloc) {
                cur->ਅੰਕੜੇ = (ਅੱਖਰ *) ਅੰਕੜੇ->ਅੰਕੜੇ + info.ਵਿੱਥ;
            }
        }

        ਜੇ (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_ਅੰਕੜੇ);
            *params.ctx = ਖਾਲੀ_ਸੂਚਕ;
            gguf_free(ctx);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ggml_ਸੈੱਟ_no_alloc(ctx_ਅੰਕੜੇ, params.no_alloc);
    }

    ਵਾਪਸ ctx;
}

ਬਣਤਰ gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, ਖਾਲੀ * userdata, size_t max_chunk_read, uint64_t max_expected_ਮਾਪ, ਬਣਤਰ gguf_init_params params) {
    ਜੇ (callback == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਅਚਲ ਬਣਤਰ gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_ਮਾਪ);
    ਵਾਪਸ gguf_init_from_reader(gr, params);
}

ਬਣਤਰ gguf_file_reader {
    FILE * file;
    uint64_t ਵਿੱਥ;
};

ਸਥਿਰ size_t gguf_file_reader_callback(ਖਾਲੀ * userdata, ਖਾਲੀ * output, uint64_t ਵਿੱਥ, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    ਜੇ (reader.ਵਿੱਥ != ਵਿੱਥ) {
        ਜੇ (ਵਿੱਥ > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(ਵਿੱਥ), SEEK_SET) != 0) {
            ਵਾਪਸ 0;
        }

        reader.ਵਿੱਥ = ਵਿੱਥ;
    }

    ਅਚਲ size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.ਵਿੱਥ += nread;
    ਵਾਪਸ nread;
}

ਬਣਤਰ gguf_context * gguf_init_from_file_ptr(FILE * file, ਬਣਤਰ gguf_init_params params) {
    ਜੇ (!file) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਅਚਲ int64_t cur = gguf_ftell(file);
    ਜੇ (cur < 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ਅਚਲ ਬਣਤਰ gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.ਵਿੱਥ, gguf_reader::file_remain(file));
    ਵਾਪਸ gguf_init_from_reader(gr, params);
}

ਬਣਤਰ gguf_ਬਫਰ_reader {
    ਅਚਲ uint8_t * ਅੰਕੜੇ;
    size_t          size;
};

ਸਥਿਰ size_t gguf_ਬਫਰ_reader_callback(ਖਾਲੀ * userdata, ਖਾਲੀ * output, uint64_t ਵਿੱਥ, size_t len) {
    GGML_ASSERT(len > 0);

    ਅਚਲ gguf_ਬਫਰ_reader & reader = *static_cast<gguf_ਬਫਰ_reader *>(userdata);

    ਜੇ (ਵਿੱਥ > reader.size || len > reader.size - ਵਿੱਥ) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ size_t ਅੰਕੜੇ_ਵਿੱਥ = static_cast<size_t>(ਵਿੱਥ);
    ਅਚਲ size_t nread = std::min(len, reader.size - ਅੰਕੜੇ_ਵਿੱਥ);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(static_cast<uint8_t *>(output), reader.ਅੰਕੜੇ + ਅੰਕੜੇ_ਵਿੱਥ, nread);
    ਵਾਪਸ nread;
}

ਬਣਤਰ gguf_context * gguf_init_from_ਬਫਰ(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size, ਬਣਤਰ gguf_init_params params) {
    ਜੇ (ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ || size == 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    gguf_ਬਫਰ_reader reader = {
        /*.data = */ static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ),
        /*.size = */ size,
    };
    ਅਚਲ ਬਣਤਰ gguf_reader gr(gguf_ਬਫਰ_reader_callback, &reader, SIZE_MAX, 0, size);
    ਵਾਪਸ gguf_init_from_reader(gr, params);
}

ਬਣਤਰ gguf_context * gguf_init_from_file(ਅਚਲ ਅੱਖਰ * fname, ਬਣਤਰ gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    ਜੇ (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਬਣਤਰ gguf_context * ਨਤੀਜਾ = gguf_init_from_file_ptr(file, params);
    fclose(file);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ gguf_free(ਬਣਤਰ gguf_context * ctx) {
    ਜੇ (ctx == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    ਮਿਟਾਓ ctx;
}

ਅਚਲ ਅੱਖਰ * gguf_type_name(ਗਿਣਤੀ gguf_type type) {
    ਸਵੈਚਾਲਿਤ it = GGUF_TYPE_NAME.find(type);
    ਵਾਪਸ it == GGUF_TYPE_NAME.end() ? ਖਾਲੀ_ਸੂਚਕ : it->second;
}

uint32_t gguf_get_ਸੰਸਕਰਣ(ਅਚਲ ਬਣਤਰ gguf_context * ctx) {
    ਵਾਪਸ ctx->ਸੰਸਕਰਣ_2;
}

size_t gguf_get_alignment(ਅਚਲ ਬਣਤਰ gguf_context * ctx) {
    ਵਾਪਸ ctx->alignment;
}

size_t gguf_get_ਅੰਕੜੇ_ਵਿੱਥ(ਅਚਲ ਬਣਤਰ gguf_context * ctx) {
    ਵਾਪਸ ctx->ਵਿੱਥ;
}

int64_t gguf_get_n_kv(ਅਚਲ ਬਣਤਰ gguf_context * ctx) {
    ਵਾਪਸ ctx->kv.size();
}

int64_t gguf_ਲੱਭੋ_ਕੁੰਜੀ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ਅਚਲ int64_t n_kv = gguf_get_n_kv(ctx);

    ਲਈ (int64_t i = 0; i < n_kv; ++i) {
        ਜੇ (strcmp(ਕੁੰਜੀ, gguf_get_ਕੁੰਜੀ(ctx, i)) == 0) {
            keyfound = i;
            ਰੋਕੋ;
        }
    }

    ਵਾਪਸ keyfound;
}

ਅਚਲ ਅੱਖਰ * gguf_get_ਕੁੰਜੀ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_ਕੁੰਜੀ().c_str();
}

ਗਿਣਤੀ gguf_type gguf_get_kv_type(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[ਕੁੰਜੀ_id].get_type();
}

ਗਿਣਤੀ gguf_type gguf_get_arr_type(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].is_array);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_type();
}

ਅਚਲ ਖਾਲੀ * gguf_get_arr_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_type() != GGUF_TYPE_STRING);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].ਅੰਕੜੇ.ਅੰਕੜੇ();
}

ਅਚਲ ਅੱਖਰ * gguf_get_arr_str(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id, size_t i) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_type() == GGUF_TYPE_STRING);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].ਅੰਕੜੇ_string[i].c_str();
}

size_t gguf_get_arr_n(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));

    ਜੇ (ctx->kv[ਕੁੰਜੀ_id].type == GGUF_TYPE_STRING) {
        ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].ਅੰਕੜੇ_string.size();
    }

    ਅਚਲ size_t type_ਮਾਪ = gguf_type_ਮਾਪ(ctx->kv[ਕੁੰਜੀ_id].type);
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].ਅੰਕੜੇ.size() % type_ਮਾਪ == 0);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].ਅੰਕੜੇ.size() / type_ਮਾਪ;
}

uint8_t gguf_get_val_u8(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<int32_t>();
}

ਦਸ਼ਮਲਵ gguf_get_val_f32(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<ਦਸ਼ਮਲਵ>();
}

uint64_t gguf_get_val_u64(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<int64_t>();
}

ਦੁੱਗਣਾ gguf_get_val_f64(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<ਦੁੱਗਣਾ>();
}

ਤਾਰਕਿਕ gguf_get_val_bool(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<ਤਾਰਕਿਕ>();
}

ਅਚਲ ਅੱਖਰ * gguf_get_val_str(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].get_val<std::string>().c_str();
}

ਅਚਲ ਖਾਲੀ * gguf_get_val_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t ਕੁੰਜੀ_id) {
    GGML_ASSERT(ਕੁੰਜੀ_id >= 0 && ਕੁੰਜੀ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[ਕੁੰਜੀ_id].get_type() != GGUF_TYPE_STRING);
    ਵਾਪਸ ctx->kv[ਕੁੰਜੀ_id].ਅੰਕੜੇ.ਅੰਕੜੇ();
}

int64_t gguf_get_n_tensors(ਅਚਲ ਬਣਤਰ gguf_context * ctx) {
    ਵਾਪਸ ctx->info.size();
}

int64_t gguf_ਲੱਭੋ_tensor(ਅਚਲ ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ਅਚਲ int64_t n_tensors = gguf_get_n_tensors(ctx);

    ਲਈ (int64_t i = 0; i < n_tensors; ++i) {
        ਜੇ (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            ਰੋਕੋ;
        }
    }

    ਵਾਪਸ tensor_id;
}

size_t gguf_get_tensor_ਵਿੱਥ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ਵਾਪਸ ctx->info[tensor_id].ਵਿੱਥ;
}

ਅਚਲ ਅੱਖਰ * gguf_get_tensor_name(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ਵਾਪਸ ctx->info[tensor_id].t.name;
}

ਅਚਲ int64_t * gguf_get_tensor_ne(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ਵਾਪਸ ctx->info[tensor_id].t.ne;
}

ਗਿਣਤੀ ggml_type gguf_get_tensor_type(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ਵਾਪਸ ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_ਮਾਪ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ਵਾਪਸ ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_ਕੁੰਜੀ(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ) {
    ਅਚਲ int64_t ਕੁੰਜੀ_id = gguf_ਲੱਭੋ_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ਜੇ (ਕੁੰਜੀ_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + ਕੁੰਜੀ_id);
    }
    ਵਾਪਸ ਕੁੰਜੀ_id;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ gguf_check_reserved_ਕੁੰਜੀਆਂ(ਅਚਲ std::string & ਕੁੰਜੀ, ਅਚਲ T val) {
    ਜੇ (ਕੁੰਜੀ == GGUF_ਕੁੰਜੀ_GENERAL_ALIGNMENT) {
        ਜੇ constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_ਕੁੰਜੀ_GENERAL_ALIGNMENT " must be power of 2");
        } ਨਹੀਂਤਾਂ {
            GGML_ਅਣਵਰਤਿਆ(val);
            GGML_ABORT(GGUF_ਕੁੰਜੀ_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_u8(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, uint8_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_i8(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, int8_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_u16(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, uint16_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_i16(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, int16_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_u32(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, uint32_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_i32(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, int32_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_f32(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, ਦਸ਼ਮਲਵ val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_u64(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, uint64_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_i64(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, int64_t val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_f64(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, ਦੁੱਗਣਾ val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_bool(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, ਤਾਰਕਿਕ val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, val);
}

ਖਾਲੀ gguf_ਸੈੱਟ_val_str(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, ਅਚਲ ਅੱਖਰ * val) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, val);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);
    ctx->kv.emplace_back(ਕੁੰਜੀ, std::string(val));
}

ਖਾਲੀ gguf_ਸੈੱਟ_arr_ਅੰਕੜੇ(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, ਗਿਣਤੀ gguf_type type, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t n) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, ਅੰਕੜੇ);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);

    ਅਚਲ size_t nbytes = n*gguf_type_ਮਾਪ(type);
    std::vector<int8_t> tmp(nbytes);
    ਜੇ (!tmp.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp.ਅੰਕੜੇ(), ਅੰਕੜੇ, nbytes);
    }
    ctx->kv.emplace_back(ਕੁੰਜੀ, tmp);
    ctx->kv.back().cast(type);
}

ਖਾਲੀ gguf_ਸੈੱਟ_arr_str(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ, ਅਚਲ ਅੱਖਰ ** ਅੰਕੜੇ, size_t n) {
    gguf_check_reserved_ਕੁੰਜੀਆਂ(ਕੁੰਜੀ, ਅੰਕੜੇ);
    gguf_remove_ਕੁੰਜੀ(ctx, ਕੁੰਜੀ);

    std::vector<std::string> tmp(n);
    ਲਈ (size_t i = 0; i < n; ++i) {
        tmp[i] = ਅੰਕੜੇ[i];
    }
    ctx->kv.emplace_back(ਕੁੰਜੀ, tmp);
}

// set or add KV pairs from another context
ਖਾਲੀ gguf_ਸੈੱਟ_kv(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਬਣਤਰ gguf_context * src) {
    ਅਚਲ int64_t n_kv = gguf_get_n_kv(src);
    ਲਈ (int64_t i = 0; i < n_kv; ++i) {
        ਅਚਲ ਬਣਤਰ gguf_kv & kv = src->kv[i];

        ਜੇ (!kv.is_array) {
            ਚੋਣ (kv.get_type()) {
                ਹਾਲਤ GGUF_TYPE_UINT8:   gguf_ਸੈੱਟ_val_u8  (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<uint8_t>());             ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT8:    gguf_ਸੈੱਟ_val_i8  (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<int8_t>());              ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_UINT16:  gguf_ਸੈੱਟ_val_u16 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<uint16_t>());            ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT16:   gguf_ਸੈੱਟ_val_i16 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<int16_t>());             ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_UINT32:  gguf_ਸੈੱਟ_val_u32 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<uint32_t>());            ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT32:   gguf_ਸੈੱਟ_val_i32 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<int32_t>());             ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_FLOAT32: gguf_ਸੈੱਟ_val_f32 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<ਦਸ਼ਮਲਵ>());               ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_UINT64:  gguf_ਸੈੱਟ_val_u64 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<uint64_t>());            ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_INT64:   gguf_ਸੈੱਟ_val_i64 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<int64_t>());             ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_FLOAT64: gguf_ਸੈੱਟ_val_f64 (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<ਦੁੱਗਣਾ>());              ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_BOOL:    gguf_ਸੈੱਟ_val_bool(ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<ਤਾਰਕਿਕ>());                ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_STRING:  gguf_ਸੈੱਟ_val_str (ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_val<std::string>().c_str()); ਰੋਕੋ;
                ਹਾਲਤ GGUF_TYPE_ARRAY:
                ਮੂਲ: GGML_ABORT("invalid type");
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t ne = kv.get_ne();

        ਚੋਣ (kv.get_type()) {
            ਹਾਲਤ GGUF_TYPE_UINT8:
            ਹਾਲਤ GGUF_TYPE_INT8:
            ਹਾਲਤ GGUF_TYPE_UINT16:
            ਹਾਲਤ GGUF_TYPE_INT16:
            ਹਾਲਤ GGUF_TYPE_UINT32:
            ਹਾਲਤ GGUF_TYPE_INT32:
            ਹਾਲਤ GGUF_TYPE_FLOAT32:
            ਹਾਲਤ GGUF_TYPE_UINT64:
            ਹਾਲਤ GGUF_TYPE_INT64:
            ਹਾਲਤ GGUF_TYPE_FLOAT64:
            ਹਾਲਤ GGUF_TYPE_BOOL: {
                gguf_ਸੈੱਟ_arr_ਅੰਕੜੇ(ctx, kv.get_ਕੁੰਜੀ().c_str(), kv.get_type(), kv.ਅੰਕੜੇ.ਅੰਕੜੇ(), ne);
            } ਰੋਕੋ;
            ਹਾਲਤ GGUF_TYPE_STRING: {
                std::vector<ਅਚਲ ਅੱਖਰ *> tmp(ne);
                ਲਈ (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.ਅੰਕੜੇ_string[j].c_str();
                }
                gguf_ਸੈੱਟ_arr_str(ctx, kv.get_ਕੁੰਜੀ().c_str(), tmp.ਅੰਕੜੇ(), ne);
            } ਰੋਕੋ;
            ਹਾਲਤ GGUF_TYPE_ARRAY:
            ਮੂਲ: GGML_ABORT("invalid type");
        }
    }
}

ਖਾਲੀ gguf_ਜੋੜੋ_tensor(
             ਬਣਤਰ gguf_context * ctx,
        ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    ਜੇ (gguf_ਲੱਭੋ_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    ਬਣਤਰ gguf_tensor_info ti;
    ti.t = *tensor;
    ti.ਵਿੱਥ = ctx->info.ਖਾਲੀ_ਥਾਵਾਂ() ? 0 :
        ctx->info.back().ਵਿੱਥ + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

ਖਾਲੀ gguf_ਸੈੱਟ_tensor_type(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * name, ਗਿਣਤੀ ggml_type type) {
    ਅਚਲ int64_t tensor_id = gguf_ਲੱਭੋ_tensor(ctx, name);
    ਜੇ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    ਬਣਤਰ ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ਅਚਲ size_t  type_ਮਾਪ = ggml_type_ਮਾਪ(type);
    ਅਚਲ int64_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_ਮਾਪ == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_ਮਾਪ;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_ਮਾਪ);
    ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ਅਚਲ int64_t n_tensors = gguf_get_n_tensors(ctx);
    ਲਈ (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].ਵਿੱਥ = ctx->info[i - 1].ਵਿੱਥ + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

ਖਾਲੀ gguf_ਸੈੱਟ_tensor_ਅੰਕੜੇ(ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * name, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ int64_t tensor_id = gguf_ਲੱਭੋ_tensor(ctx, name);
    ਜੇ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.ਅੰਕੜੇ = (ਖਾਲੀ *)(uintptr_t)ਅੰਕੜੇ; // double cast suppresses warning about casting away const
}

ਬਣਤਰ gguf_writer_ਅਧਾਰ {
    size_t written_ਬਾਈਟਾਂ {0u};

    ਆਭਾਸੀ ~gguf_writer_ਅਧਾਰ(ਖਾਲੀ) = ਮੂਲ;

    // we bet on devirtualization
    ਆਭਾਸੀ ਖਾਲੀ ਲਿਖ(int8_t val) = 0;
    ਆਭਾਸੀ ਖਾਲੀ ਲਿਖ(ਅਚਲ std::vector<int8_t> & val) = 0;
    ਆਭਾਸੀ ਖਾਲੀ write_tensor_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ gguf_tensor_info & info, size_t ਵਿੱਥ_ਅੰਕੜੇ, size_t alignment) = 0;

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਖਾਲੀ ਲਿਖ(ਅਚਲ T & val) {
        ਲਈ (size_t i = 0; i < ਆਕਾਰ(val); ++i) {
            ਲਿਖ(reinterpret_cast<ਅਚਲ int8_t *>(&val)[i]);
        }
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ ਤਾਰਕਿਕ & val) {
        ਅਚਲ int8_t val8 = val ? 1 : 0;
        ਲਿਖ(val8);
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ std::string & val) {
        {
            ਅਚਲ uint64_t n = val.ਲੰਮਾਈ();
            ਲਿਖ(n);
        }
        ਲਈ (size_t i = 0; i < val.ਲੰਮਾਈ(); ++i) {
            ਲਿਖ((val.ਅੰਕੜੇ())[i]);
        }
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ ਅੱਖਰ * val) {
        ਲਿਖ(std::string(val));
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ ਗਿਣਤੀ ggml_type & val) {
        ਲਿਖ(int32_t(val));
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ ਗਿਣਤੀ gguf_type & val) {
        ਲਿਖ(int32_t(val));
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ ਬਣਤਰ gguf_kv & kv) {
        ਅਚਲ uint64_t ne = kv.get_ne();

        ਲਿਖ(kv.get_ਕੁੰਜੀ());

        ਜੇ (kv.is_array) {
            ਲਿਖ(GGUF_TYPE_ARRAY);
            ਲਿਖ(kv.get_type());
            ਲਿਖ(ne);
        } ਨਹੀਂਤਾਂ {
            ਲਿਖ(kv.get_type());
        }

        ਚੋਣ (kv.get_type()) {
            ਹਾਲਤ GGUF_TYPE_UINT8:
            ਹਾਲਤ GGUF_TYPE_INT8:
            ਹਾਲਤ GGUF_TYPE_UINT16:
            ਹਾਲਤ GGUF_TYPE_INT16:
            ਹਾਲਤ GGUF_TYPE_UINT32:
            ਹਾਲਤ GGUF_TYPE_INT32:
            ਹਾਲਤ GGUF_TYPE_FLOAT32:
            ਹਾਲਤ GGUF_TYPE_UINT64:
            ਹਾਲਤ GGUF_TYPE_INT64:
            ਹਾਲਤ GGUF_TYPE_FLOAT64: {
                ਲਿਖ(kv.ਅੰਕੜੇ);
            } ਰੋਕੋ;
            ਹਾਲਤ GGUF_TYPE_BOOL: {
                ਲਈ (size_t i = 0; i < ne; ++i) {
                    ਲਿਖ(kv.get_val<ਤਾਰਕਿਕ>(i));
                }
            } ਰੋਕੋ;
            ਹਾਲਤ GGUF_TYPE_STRING: {
                ਲਈ (size_t i = 0; i < ne; ++i) {
                    ਲਿਖ(kv.get_val<std::string>(i));
                }
            } ਰੋਕੋ;
            ਹਾਲਤ GGUF_TYPE_ARRAY:
            ਮੂਲ: GGML_ABORT("invalid type");
        }
    }

    ਖਾਲੀ write_tensor_meta(ਅਚਲ ਬਣਤਰ gguf_tensor_info & info) {
        ਲਿਖ(info.t.name);

        ਅਚਲ uint32_t n_dims = ggml_n_dims(&info.t);
        ਲਿਖ(n_dims);

        ਲਈ (uint32_t j = 0; j < n_dims; ++j) {
            ਲਿਖ(info.t.ne[j]);
        }
        ਲਿਖ(info.t.type);
        ਲਿਖ(info.ਵਿੱਥ);
    }

    ਖਾਲੀ pad(ਅਚਲ size_t alignment) {
        ਜਦੋਂ (written_ਬਾਈਟਾਂ % alignment != 0) {
            ਅਚਲ int8_t zero = 0;
            ਲਿਖ(zero);
        }
    }
};

// vector buffer based writer
ਬਣਤਰ gguf_writer_buf final : ਜਨਤਕ gguf_writer_ਅਧਾਰ {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    ਵਰਤੋਂ gguf_writer_ਅਧਾਰ::ਲਿਖ;

    ਖਾਲੀ ਲਿਖ(ਅਚਲ int8_t val) override {
        buf.push_back(val);
        written_ਬਾਈਟਾਂ++;
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_ਬਾਈਟਾਂ += val.size();
    }

    ਖਾਲੀ write_tensor_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ gguf_tensor_info & info, ਅਚਲ size_t ਵਿੱਥ_ਅੰਕੜੇ, ਅਚਲ size_t alignment) override {
        GGML_ASSERT(buf.size() - ਵਿੱਥ_ਅੰਕੜੇ == info.ਵਿੱਥ);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ਅਚਲ size_t ਵਿੱਥ = buf.size();
        ਅਚਲ size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(ਵਿੱਥ + nbytes);
        ਜੇ (info.t.ਬਫਰ) {
            ggml_backend_tensor_get(&info.t, buf.ਅੰਕੜੇ() + ਵਿੱਥ, 0, nbytes);
        } ਨਹੀਂਤਾਂ {
            GGML_ASSERT(info.t.ਅੰਕੜੇ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(buf.ਅੰਕੜੇ() + ਵਿੱਥ, info.t.ਅੰਕੜੇ, nbytes);
        }
        written_ਬਾਈਟਾਂ += nbytes;

        pad(alignment);
    }
};

// file based writer
ਬਣਤਰ gguf_writer_file final : ਜਨਤਕ gguf_writer_ਅਧਾਰ {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    ਵਰਤੋਂ gguf_writer_ਅਧਾਰ::ਲਿਖ;

    ਖਾਲੀ ਲਿਖ(ਅਚਲ int8_t val) override {
        ਅਚਲ ਸਵੈਚਾਲਿਤ real_val = static_cast<uint8_t>(val);
        ਅਚਲ ਸਵੈਚਾਲਿਤ ret = fputc(real_val, file);
        written_ਬਾਈਟਾਂ++;
        ਜੇ (ret != real_val) {
            ਸੁੱਟੋ std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((ਪੂਰਨ)real_val) + "'");
        }
    }

    ਖਾਲੀ ਲਿਖ(ਅਚਲ std::vector<int8_t> & val) override {
        ਅਚਲ ਸਵੈਚਾਲਿਤ ret = fwrite(val.ਅੰਕੜੇ(), 1, val.size(), file);
        written_ਬਾਈਟਾਂ += val.size();
        ਜੇ (ret != val.size()) {
            ਸੁੱਟੋ std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    ਖਾਲੀ write_tensor_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ gguf_tensor_info & info, ਅਚਲ size_t ਵਿੱਥ_ਅੰਕੜੇ, ਅਚਲ size_t alignment) override {
        GGML_ASSERT(written_ਬਾਈਟਾਂ - ਵਿੱਥ_ਅੰਕੜੇ == info.ਵਿੱਥ);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ਅਚਲ size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        ਜੇ (info.t.ਬਫਰ) {
            ggml_backend_tensor_get(&info.t, buf.ਅੰਕੜੇ(), 0, nbytes);
        } ਨਹੀਂਤਾਂ {
            GGML_ASSERT(info.t.ਅੰਕੜੇ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(buf.ਅੰਕੜੇ(), info.t.ਅੰਕੜੇ, nbytes);
        }
        ਲਿਖ(buf);

        pad(alignment);
    }
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ writer_t>
ਸਥਿਰ ਖਾਲੀ gguf_write_ਬਾਹਰਲਾ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, writer_t & gw, ਤਾਰਕਿਕ only_meta) {
    ਅਚਲ int64_t n_kv      = gguf_get_n_kv(ctx);
    ਅਚਲ int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.ਲਿਖ(GGUF_MAGIC[0]);
    gw.ਲਿਖ(GGUF_MAGIC[1]);
    gw.ਲਿਖ(GGUF_MAGIC[2]);
    gw.ਲਿਖ(GGUF_MAGIC[3]);
    gw.ਲਿਖ(ctx->ਸੰਸਕਰਣ_2);
    gw.ਲਿਖ(n_tensors);
    gw.ਲਿਖ(n_kv);

    // write key-value pairs
    ਲਈ (int64_t i = 0; i < n_kv; ++i) {
        gw.ਲਿਖ(ctx->kv[i]);
    }

    // write tensor info
    ਲਈ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    ਜੇ (only_meta) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t ਵਿੱਥ_ਅੰਕੜੇ = gw.written_ਬਾਈਟਾਂ;

    // write tensor data
    ਲਈ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_ਅੰਕੜੇ(ctx->info[i], ਵਿੱਥ_ਅੰਕੜੇ, ctx->alignment);
    }
}

ਖਾਲੀ gguf_write_ਮੰਜ਼ਿਲ_buf(ਅਚਲ ਬਣਤਰ gguf_context * ctx, std::vector<int8_t> & buf, ਤਾਰਕਿਕ only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_ਬਾਹਰਲਾ(ctx, gw, only_meta);
}

ਤਾਰਕਿਕ gguf_write_ਮੰਜ਼ਿਲ_file_ptr(ਅਚਲ ਬਣਤਰ gguf_context * ctx, FILE * file, ਤਾਰਕਿਕ only_meta) {
    GGML_ASSERT(file);

    ਕੋਸ਼ਿਸ਼ {
        gguf_writer_file gw(file);
        gguf_write_ਬਾਹਰਲਾ(ctx, gw, only_meta);
    } ਫੜੋ (ਅਚਲ std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ gguf_write_ਮੰਜ਼ਿਲ_file(ਅਚਲ ਬਣਤਰ gguf_context * ctx, ਅਚਲ ਅੱਖਰ * fname, ਤਾਰਕਿਕ only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    ਜੇ (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ success = gguf_write_ਮੰਜ਼ਿਲ_file_ptr(ctx, file, only_meta);
    ਜੇ (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    ਵਾਪਸ success;
}

size_t gguf_get_meta_ਮਾਪ(ਅਚਲ ਬਣਤਰ gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_ਮੰਜ਼ਿਲ_buf(ctx, buf, /*only_meta =*/ ਸੱਚ);
    ਵਾਪਸ buf.size();
}

ਖਾਲੀ gguf_get_meta_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ gguf_context * ctx, ਖਾਲੀ * ਅੰਕੜੇ) {
    std::vector<int8_t> buf;
    gguf_write_ਮੰਜ਼ਿਲ_buf(ctx, buf, /*only_meta =*/ ਸੱਚ);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, buf.ਅੰਕੜੇ(), buf.size());
}
