#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "gguf.h"

#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <new>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿಸು GGUF_MAX_STRING_ಉದ್ದಳತೆ  (1024*1024*1024)
#ವ್ಯಾಖ್ಯಾನಿಸು GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#ಇಲ್ಲದಿದ್ದರೆ
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#ಆದರೆ_ಅಂತ್ಯ

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ರಚನೆ type_ಗಮ್ಯ_gguf_type;

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<uint8_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_UINT8;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<int8_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_INT8;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<uint16_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_UINT16;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<int16_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_INT16;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<uint32_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_UINT32;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<int32_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_INT32;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<ದಶಮಾಂಶ> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_FLOAT32;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<ತಾರ್ಕಿಕ> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_BOOL;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<std::string> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_STRING;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<uint64_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_UINT64;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<int64_t> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_INT64;
};

ಮಾದರಿ <>
ರಚನೆ type_ಗಮ್ಯ_gguf_type<ದ್ವಿಗುಣ> {
    ಸ್ಥಿರ constexpr ಎಣಿಕೆ gguf_type value = GGUF_TYPE_FLOAT64;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::map<gguf_type, size_t> GGUF_TYPE_ಗಾತ್ರ = {
    {GGUF_TYPE_UINT8,   ಗಾತ್ರ(uint8_t)},
    {GGUF_TYPE_INT8,    ಗಾತ್ರ(int8_t)},
    {GGUF_TYPE_UINT16,  ಗಾತ್ರ(uint16_t)},
    {GGUF_TYPE_INT16,   ಗಾತ್ರ(int16_t)},
    {GGUF_TYPE_UINT32,  ಗಾತ್ರ(uint32_t)},
    {GGUF_TYPE_INT32,   ಗಾತ್ರ(int32_t)},
    {GGUF_TYPE_FLOAT32, ಗಾತ್ರ(ದಶಮಾಂಶ)},
    {GGUF_TYPE_BOOL,    ಗಾತ್ರ(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  ಗಾತ್ರ(uint64_t)},
    {GGUF_TYPE_INT64,   ಗಾತ್ರ(int64_t)},
    {GGUF_TYPE_FLOAT64, ಗಾತ್ರ(ದ್ವಿಗುಣ)},
};
static_assert(GGUF_TYPE_ಎಣಿಕೆ == 13, "GGUF_TYPE_COUNT != 13");

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::map<gguf_type, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *> GGUF_TYPE_NAME = {
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
static_assert(GGUF_TYPE_ಎಣಿಕೆ == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_ಗಾತ್ರ(ಎಣಿಕೆ gguf_type type) {
    ಸ್ವಯಂಚಾಲಿತ it = GGUF_TYPE_ಗಾತ್ರ.find(type);
    ಹಿಂತಿರುಗಿ it == GGUF_TYPE_ಗಾತ್ರ.end() ? 0 : it->second;
}

ರಚನೆ gguf_kv {
    std::string ಕೀಲಿ;

    ತಾರ್ಕಿಕ is_array;
    ಎಣಿಕೆ gguf_type type;

    std::vector<int8_t>      ದತ್ತಾಂಶ;
    std::vector<std::string> ದತ್ತಾಂಶ_string;

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    gguf_kv(ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ, ಸ್ಥಿರಾಂಕ T value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ಸುಳ್ಳು), type(type_ಗಮ್ಯ_gguf_type<T>::value) {
        GGML_ASSERT(!ಕೀಲಿ.ಖಾಲಿಗಳು());
        ದತ್ತಾಂಶ.resize(ಗಾತ್ರ(T));
        ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), &value, ಗಾತ್ರ(T));
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    gguf_kv(ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<T> & value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ನಿಜ), type(type_ಗಮ್ಯ_gguf_type<T>::value) {
        GGML_ASSERT(!ಕೀಲಿ.ಖಾಲಿಗಳು());
        ದತ್ತಾಂಶ.resize(value.size()*ಗಾತ್ರ(T));
        ಗಾಗಿ (size_t i = 0; i < value.size(); ++i) {
            ಸ್ಥಿರಾಂಕ T tmp = value[i];
            ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ.ದತ್ತಾಂಶ() + i*ಗಾತ್ರ(T), &tmp, ಗಾತ್ರ(T));
        }
    }

    gguf_kv(ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::string & value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ಸುಳ್ಳು), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ಕೀಲಿ.ಖಾಲಿಗಳು());
        ದತ್ತಾಂಶ_string.push_back(value);
    }

    gguf_kv(ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<std::string> & value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ನಿಜ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ಕೀಲಿ.ಖಾಲಿಗಳು());
        ದತ್ತಾಂಶ_string = value;
    }

    ಸ್ಥಿರಾಂಕ std::string & get_ಕೀಲಿ() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ಕೀಲಿ;
    }

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ gguf_type & get_type() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type;
    }

    size_t get_ne() ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (type == GGUF_TYPE_STRING) {
            ಸ್ಥಿರಾಂಕ size_t ne = ದತ್ತಾಂಶ_string.size();
            GGML_ASSERT(is_array || ne == 1);
            ಹಿಂತಿರುಗಿ ne;
        }
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = gguf_type_ಗಾತ್ರ(type);
        GGML_ASSERT(ದತ್ತಾಂಶ.size() % type_ಗಾತ್ರ == 0);
        ಸ್ಥಿರಾಂಕ size_t ne = ದತ್ತಾಂಶ.size() / type_ಗಾತ್ರ;
        GGML_ASSERT(is_array || ne == 1);
        ಹಿಂತಿರುಗಿ ne;
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ಸ್ಥಿರಾಂಕ T & get_val(ಸ್ಥಿರಾಂಕ size_t i = 0) ಸ್ಥಿರಾಂಕ {
        GGML_ASSERT(type_ಗಮ್ಯ_gguf_type<T>::value == type);
        ಆದರೆ constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(ದತ್ತಾಂಶ_string.size() >= i+1);
            ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ_string[i];
        }
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = gguf_type_ಗಾತ್ರ(type);
        GGML_ASSERT(ದತ್ತಾಂಶ.size() % type_ಗಾತ್ರ == 0);
        GGML_ASSERT(ದತ್ತಾಂಶ.size() >= (i+1)*type_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ reinterpret_cast<ಸ್ಥಿರಾಂಕ T *>(ದತ್ತಾಂಶ.ದತ್ತಾಂಶ())[i];
    }

    ಶೂನ್ಯ cast(ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ gguf_type new_type) {
        ಸ್ಥಿರಾಂಕ size_t new_type_ಗಾತ್ರ = gguf_type_ಗಾತ್ರ(new_type);
        GGML_ASSERT(ದತ್ತಾಂಶ.size() % new_type_ಗಾತ್ರ == 0);
        type = new_type;
    }
};

ರಚನೆ gguf_tensor_info {
    ರಚನೆ ggml_tensor t; // for holding the equivalent info
    uint64_t ಸ್ಥಾನಾಂತರ;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

ರಚನೆ gguf_context {
    uint32_t ಆವೃತ್ತಿ_2 = GGUF_ಆವೃತ್ತಿ;

    std::vector<ರಚನೆ gguf_kv> kv;
    std::vector<ರಚನೆ gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t ಸ್ಥಾನಾಂತರ    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    ಶೂನ್ಯ * ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
};

ರಚನೆ gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            ಶೂನ್ಯ * userdata,
            size_t max_chunk_read,
            uint64_t ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ = 0,
            uint64_t nbytes_remain = 0)
        : callback(callback),
          userdata(userdata),
          max_chunk_read(max_chunk_read),
          ಆರಂಭ_ಸ್ಥಾನಾಂತರ(ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ),
          ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ(ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ),
          nbytes_remain(nbytes_remain) {
        GGML_ASSERT(max_chunk_read > 0);
    }

    // helper for remaining bytes in a file
    ಸ್ಥಿರ uint64_t file_remain(FILE * file) {
        ಸ್ಥಿರಾಂಕ int64_t cur = gguf_ftell(file);
        ಆದರೆ (cur < 0) {
            ಹಿಂತಿರುಗಿ 0;
        }
        ಆದರೆ (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ಹಿಂತಿರುಗಿ 0;
        }
        ಸ್ಥಿರಾಂಕ int64_t end = gguf_ftell(file);
        ಆದರೆ (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ಹಿಂತಿರುಗಿ 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        ಹಿಂತಿರುಗಿ static_cast<uint64_t>(end - cur);
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ತಾರ್ಕಿಕ ಪಠ(T & dst) ಸ್ಥಿರಾಂಕ {
        ಸ್ಥಿರಾಂಕ size_t size = ಗಾತ್ರ(dst);
        ಆದರೆ (size > nbytes_remain) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ read_raw(&dst, size) == size;
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ತಾರ್ಕಿಕ ಪಠ(std::vector<T> & dst, ಸ್ಥಿರಾಂಕ size_t n) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (n > GGUF_MAX_ARRAY_ELEMENTS) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            ಆದರೆ (n > SIZE_MAX / ಗಾತ್ರ(uint64_t)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (nbytes_remain < n * ಗಾತ್ರ(uint64_t)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (n > SIZE_MAX / ಗಾತ್ರ(T)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (nbytes_remain < n * ಗಾತ್ರ(T)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        dst.resize(n);
        ಗಾಗಿ (size_t i = 0; i < dst.size(); ++i) {
            ಆದರೆ constexpr (std::is_same<T, ತಾರ್ಕಿಕ>::value) {
                ತಾರ್ಕಿಕ tmp;
                ಆದರೆ (!ಪಠ(tmp)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                dst[i] = tmp;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (!ಪಠ(dst[i])) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ ಪಠ(ತಾರ್ಕಿಕ & dst) ಸ್ಥಿರಾಂಕ {
        int8_t tmp = -1;
        ಆದರೆ (!ಪಠ(tmp)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        dst = tmp != 0;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ ಪಠ(ಎಣಿಕೆ ggml_type & dst) ಸ್ಥಿರಾಂಕ {
        int32_t tmp = -1;
        ಆದರೆ (!ಪಠ(tmp)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        dst = ggml_type(tmp);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ ಪಠ(ಎಣಿಕೆ gguf_type & dst) ಸ್ಥಿರಾಂಕ {
        int32_t tmp = -1;
        ಆದರೆ (!ಪಠ(tmp)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        dst = gguf_type(tmp);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ ಪಠ(std::string & dst) ಸ್ಥಿರಾಂಕ {
        uint64_t size = 0;
        ಆದರೆ (!ಪಠ(size)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (size > GGUF_MAX_STRING_ಉದ್ದಳತೆ) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_ಉದ್ದಳತೆ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        dst.resize(static_cast<size_t>(size));
        ಹಿಂತಿರುಗಿ read_raw(dst.ದತ್ತಾಂಶ(), static_cast<size_t>(size)) == size;
    }

    ತಾರ್ಕಿಕ ಪಠ(ಶೂನ್ಯ * dst, ಸ್ಥಿರಾಂಕ size_t size) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (size > nbytes_remain) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ read_raw(dst, size) == size;
    }

    uint64_t tell() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ಆರಂಭ_ಸ್ಥಾನಾಂತರ;
    }

    ತಾರ್ಕಿಕ seek(uint64_t absolute_ಸ್ಥಾನಾಂತರ) ಸ್ಥಿರಾಂಕ {
        ಸ್ಥಿರಾಂಕ uint64_t ಕೊನೆ_ಸ್ಥಾನಾಂತರ = uint64_t(ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ) + nbytes_remain;
        ಆದರೆ (absolute_ಸ್ಥಾನಾಂತರ > ಕೊನೆ_ಸ್ಥಾನಾಂತರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ = absolute_ಸ್ಥಾನಾಂತರ;
        nbytes_remain = ಕೊನೆ_ಸ್ಥಾನಾಂತರ - absolute_ಸ್ಥಾನಾಂತರ;

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

ಖಾಸಗಿ:
    size_t read_raw(ಶೂನ್ಯ * dst, size_t size) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (callback == ಶೂನ್ಯಸೂಚಕ || size == 0) {
            ಹಿಂತಿರುಗಿ 0;
        }

        uint8_t * ದತ್ತಾಂಶ = static_cast<uint8_t *>(dst);
        size_t ಒಟ್ಟು_nread = 0;
        ತಾರ್ಕಿಕ reached_eof = ಸುಳ್ಳು;

        ಆಗ (ಒಟ್ಟು_nread < size) {
            ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ = std::min(max_chunk_read, size - ಒಟ್ಟು_nread);
            ಆದರೆ (ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ + ಒಟ್ಟು_nread < ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ) {
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿರಾಂಕ size_t nread = callback(userdata, static_cast<ಶೂನ್ಯ *>(ದತ್ತಾಂಶ + ಒಟ್ಟು_nread), ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ + ಒಟ್ಟು_nread, chunk_ಗಾತ್ರ);
            ಒಟ್ಟು_nread += nread;
            ಆದರೆ (nread != chunk_ಗಾತ್ರ) {
                reached_eof = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        }

        ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ += ಒಟ್ಟು_nread;
        GGML_ASSERT(ಒಟ್ಟು_nread <= nbytes_remain);
        nbytes_remain -= ಒಟ್ಟು_nread;

        ಆದರೆ (reached_eof) {
            nbytes_remain = 0;
        }

        ಹಿಂತಿರುಗಿ ಒಟ್ಟು_nread;
    }

    gguf_reader_callback_t callback = ಶೂನ್ಯಸೂಚಕ;
    ಶೂನ್ಯ * userdata = ಶೂನ್ಯಸೂಚಕ;
    size_t max_chunk_read = 0;
    uint64_t ಆರಂಭ_ಸ್ಥಾನಾಂತರ = 0;
    ಬದಲಾಯಿಸಬಹುದಾದ uint64_t ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ = 0;
    ಬದಲಾಯಿಸಬಹುದಾದ uint64_t nbytes_remain = 0;
};

ರಚನೆ gguf_context * gguf_init_ಖಾಲಿಗಳು(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ ಹೊಸ gguf_context;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ತಾರ್ಕಿಕ gguf_read_emplace_helper(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_reader & gr, std::vector<ರಚನೆ gguf_kv> & kv, ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_array, ಸ್ಥಿರಾಂಕ size_t n) {
    ಆದರೆ (is_array) {
        std::vector<T> value;
        ಪ್ರಯತ್ನಿಸು {
            ಆದರೆ (!gr.ಪಠ(value, n)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        } ಹಿಡಿ (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, ಕೀಲಿ.c_str());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        } ಹಿಡಿ (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, ಕೀಲಿ.c_str());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        kv.emplace_back(ಕೀಲಿ, value);
    } ಇಲ್ಲದಿದ್ದರೆ {
        T value;
        ಆದರೆ (!gr.ಪಠ(value)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        kv.emplace_back(ಕೀಲಿ, value);
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ರಚನೆ gguf_context * gguf_init_from_reader(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_reader & gr, ರಚನೆ gguf_init_params params) {
    ರಚನೆ gguf_context * ctx = ಹೊಸ gguf_context;

    ತಾರ್ಕಿಕ ok = ನಿಜ;

    // file magic
    {
        std::vector<ಅಕ್ಷರ> magic;
        ok = ok && gr.ಪಠ(magic, 4);

        ಆದರೆ (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ಗಾಗಿ (uint32_t i = 0; i < magic.size(); i++) {
            ಆದರೆ (magic[i] != GGUF_MAGIC[i]) {
                ಅಕ್ಷರ c0 = isprint(magic[0]) ? magic[0] : '?';
                ಅಕ್ಷರ c1 = isprint(magic[1]) ? magic[1] : '?';
                ಅಕ್ಷರ c2 = isprint(magic[2]) ? magic[2] : '?';
                ಅಕ್ಷರ c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }
    }

    //ಶೀರ್ಷಿಕೆ
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    ಆದರೆ (ok && gr.ಪಠ(ctx->ಆವೃತ್ತಿ_2)) {
        ಆದರೆ (ok && ctx->ಆವೃತ್ತಿ_2 == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->ಆವೃತ್ತಿ_2);
            ok = ಸುಳ್ಳು;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        ಆದರೆ (ok && (ctx->ಆವೃತ್ತಿ_2 & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->ಆವೃತ್ತಿ_2);
            ok = ಸುಳ್ಳು;
        }

        ಆದರೆ (ok && ctx->ಆವೃತ್ತಿ_2 == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = ಸುಳ್ಳು;
        }
        ಆದರೆ (ok && ctx->ಆವೃತ್ತಿ_2 > GGUF_ಆವೃತ್ತಿ) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->ಆವೃತ್ತಿ_2, GGUF_ಆವೃತ್ತಿ);
            ok = ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ok = ಸುಳ್ಳು;
    }

    ಆದರೆ (ok && gr.ಪಠ(n_tensors)) {
        static_assert(ಗಾತ್ರ(size_t) <= 8 && ಗಾತ್ರ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ಆದರೆ (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/ಗಾತ್ರ(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/ಗಾತ್ರ(gguf_tensor_info));
            ok = ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ok = ಸುಳ್ಳು;
    }

    ಆದರೆ (ok && gr.ಪಠ(n_kv)) {
        static_assert(ಗಾತ್ರ(size_t) <= 8 && ಗಾತ್ರ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ಆದರೆ (n_kv < 0 || n_kv > int64_t(SIZE_MAX/ಗಾತ್ರ(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/ಗಾತ್ರ(gguf_kv));
            ok = ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ok = ಸುಳ್ಳು;
    }

    ಆದರೆ (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    // KV pairs
    {
        std::unordered_set<std::string> ನೋಡಿದ_ಕೀಲಿಗಳು;

        ಗಾಗಿ (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string ಕೀಲಿ;
            gguf_type   type     = gguf_type(-1);
            ತಾರ್ಕಿಕ        is_array = ಸುಳ್ಳು;
            uint64_t    n        = 1;

            ಪ್ರಯತ್ನಿಸು {
                ok = ok && gr.ಪಠ(ಕೀಲಿ);
            } ಹಿಡಿ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = ಸುಳ್ಳು;
            } ಹಿಡಿ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = ಸುಳ್ಳು;
            }
            ಆದರೆ (ok && ಕೀಲಿ.ಖಾಲಿಗಳು()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = ಸುಳ್ಳು;
            }
            ಆದರೆ (ok && !ನೋಡಿದ_ಕೀಲಿಗಳು.insert(ಕೀಲಿ).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, ಕೀಲಿ.c_str(), i);
                ok = ಸುಳ್ಳು;
            }
            ಆದರೆ (!ok) {
                ನಿಲ್ಲಿಸು;
            }

            ok = ok && gr.ಪಠ(type);
            ಆದರೆ (type == GGUF_TYPE_ARRAY) {
                is_array = ನಿಜ;
                ok = ok && gr.ಪಠ(type);
                ok = ok && gr.ಪಠ(n);
            }
            ಆದರೆ (!ok) {
                ನಿಲ್ಲಿಸು;
            }

            ಆಯ್ಕೆ (type) {
                ಸ್ಥಿತಿ GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<ದಶಮಾಂಶ>      (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<ತಾರ್ಕಿಕ>       (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<ದ್ವಿಗುಣ>     (gr, ctx->kv, ಕೀಲಿ, is_array, n); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_ARRAY:
                ಪೂರ್ವನಿಯೋಜಿತ:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, ಕೀಲಿ.c_str(), type);
                        ok = ಸುಳ್ಳು;
                    } ನಿಲ್ಲಿಸು;
            }
        }

        ಆದರೆ (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ alignment_idx = gguf_ಕಂಡುಹಿಡಿ_ಕೀಲಿ(ctx, GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT);
        ಆದರೆ (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        ಆದರೆ (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> ನೋಡಿದ_tensor_ಹೆಸರುಗಳು;

    ಗಾಗಿ (int64_t i = 0; ok && i < n_tensors; ++i) {
        ರಚನೆ gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            ಪ್ರಯತ್ನಿಸು {
                ok = ok && gr.ಪಠ(name);
            } ಹಿಡಿ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ಸುಳ್ಳು;
            } ಹಿಡಿ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ಸುಳ್ಳು;
            }
            ಆದರೆ (name.ಉದ್ದಳತೆ() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.ಉದ್ದಳತೆ(), GGML_MAX_NAME);
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
            ggml_ಗಣ_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            ಆದರೆ (ok && !ನೋಡಿದ_tensor_ಹೆಸರುಗಳು.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (!ok) {
            ನಿಲ್ಲಿಸು;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.ಪಠ(n_dims);
            ಆದರೆ (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
            ಗಾಗಿ (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                ಆದರೆ (j < n_dims) {
                    ok = ok && gr.ಪಠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                ಆದರೆ (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            ಆದರೆ (ok) {
                int64_t current_nelements = info.t.ne[0];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                    ಆದರೆ (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        ಆದರೆ (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = ಸುಳ್ಳು;
                             ನಿಲ್ಲಿಸು;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 ಆದರೆ (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     ನಿಲ್ಲಿಸು;
                 }
            }
        }
        ಆದರೆ (!ok) {
            ನಿಲ್ಲಿಸು;
        }

        // tensor type
        {
            ok = ok && gr.ಪಠ(info.t.type);

            // check that tensor type is within defined range
            ಆದರೆ (info.t.type < 0 || info.t.type >= GGML_TYPE_ಎಣಿಕೆ) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_ಎಣಿಕೆ);
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿರಾಂಕ size_t  type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(info.t.type);
            ಸ್ಥಿರಾಂಕ int64_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(info.t.type);

            // check that row size is divisible by block size
            ಆದರೆ (blck_ಗಾತ್ರ == 0 || info.t.ne[0] % blck_ಗಾತ್ರ != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (ಪೂರ್ಣಾಂಕ) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_ಗಾತ್ರ);
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }

            // check that the size of the tensor in bytes is representable
            ಆದರೆ (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_ಗಾತ್ರ(info.t.type)) > SIZE_MAX/ggml_type_ಗಾತ್ರ(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_ಗಾತ್ರ;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_ಗಾತ್ರ);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        ಆದರೆ (!ok) {
            ನಿಲ್ಲಿಸು;
        }

        // tensor data offset within buffer
        ok = ok && gr.ಪಠ(info.ಸ್ಥಾನಾಂತರ);

        ctx->info.push_back(info);
    }

    ಆದರೆ (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    ಆದರೆ (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    // store the current file offset - this is where the data section starts
    ctx->ಸ್ಥಾನಾಂತರ = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        ಗಾಗಿ (size_t i = 0; i < ctx->info.size(); ++i) {
            ಸ್ಥಿರಾಂಕ gguf_tensor_info & ti = ctx->info[i];
            ಆದರೆ (ti.ಸ್ಥಾನಾಂತರ != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.ಸ್ಥಾನಾಂತರ, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ಸ್ಥಿರಾಂಕ size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            ಆದರೆ (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ಸ್ಥಿರಾಂಕ size_t padded_ಗಾತ್ರ = GGML_PAD(nbytes, ctx->alignment);
            ಆದರೆ (SIZE_MAX - ctx->size < padded_ಗಾತ್ರ) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_ಗಾತ್ರ);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ctx->size += padded_ಗಾತ್ರ;
        }
    }

    // load the tensor data only if requested
    ಆದರೆ (params.ctx != ಶೂನ್ಯಸೂಚಕ) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_ಗಾತ್ರ = 0;
        ಆದರೆ (params.no_alloc) {
            ಆದರೆ (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }

            ಸ್ಥಿರಾಂಕ size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_ಗಾತ್ರ = overhead;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }

            ಸ್ಥಿರಾಂಕ size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            ಆದರೆ (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }

            mem_ಗಾತ್ರ = overhead + ctx->size;
        }

        ರಚನೆ ggml_init_params pdata = {
            /*mem_size   =*/ mem_ಗಾತ್ರ,
            /*mem_buffer =*/ ಶೂನ್ಯಸೂಚಕ,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        ಆದರೆ (*params.ctx == ಶೂನ್ಯಸೂಚಕ) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ರಚನೆ ggml_context * ctx_ದತ್ತಾಂಶ = *params.ctx;

        ರಚನೆ ggml_tensor * ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;

        ಆದರೆ (!params.no_alloc) {
            ದತ್ತಾಂಶ = ggml_new_tensor_1d(ctx_ದತ್ತಾಂಶ, GGML_TYPE_I8, ctx->size);

            ok = ok && ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ;

            ಆದರೆ (ok) {
                ggml_ಗಣ_name(ದತ್ತಾಂಶ, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.ಪಠ(ದತ್ತಾಂಶ->ದತ್ತಾಂಶ, ctx->size);

            ಆದರೆ (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_ದತ್ತಾಂಶ);
                *params.ctx = ಶೂನ್ಯಸೂಚಕ;
                gguf_free(ctx);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }

            ctx->ದತ್ತಾಂಶ = ದತ್ತಾಂಶ->ದತ್ತಾಂಶ;
        }

        ggml_ಗಣ_no_alloc(ctx_ದತ್ತಾಂಶ, ನಿಜ);

        // create the tensors
        ಗಾಗಿ (size_t i = 0; i < ctx->info.size(); ++i) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_tensor_info & info = ctx->info[i];

            ರಚನೆ ggml_tensor * cur = ggml_new_tensor(ctx_ದತ್ತಾಂಶ, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != ಶೂನ್ಯಸೂಚಕ;

            ಆದರೆ (!ok) {
                ನಿಲ್ಲಿಸು;
            }

            ggml_ಗಣ_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            ಆದರೆ (!params.no_alloc) {
                cur->ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) ದತ್ತಾಂಶ->ದತ್ತಾಂಶ + info.ಸ್ಥಾನಾಂತರ;
            }
        }

        ಆದರೆ (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_ದತ್ತಾಂಶ);
            *params.ctx = ಶೂನ್ಯಸೂಚಕ;
            gguf_free(ctx);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ggml_ಗಣ_no_alloc(ctx_ದತ್ತಾಂಶ, params.no_alloc);
    }

    ಹಿಂತಿರುಗಿ ctx;
}

ರಚನೆ gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, ಶೂನ್ಯ * userdata, size_t max_chunk_read, uint64_t max_expected_ಗಾತ್ರ, ರಚನೆ gguf_init_params params) {
    ಆದರೆ (callback == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_ಗಾತ್ರ);
    ಹಿಂತಿರುಗಿ gguf_init_from_reader(gr, params);
}

ರಚನೆ gguf_file_reader {
    FILE * file;
    uint64_t ಸ್ಥಾನಾಂತರ;
};

ಸ್ಥಿರ size_t gguf_file_reader_callback(ಶೂನ್ಯ * userdata, ಶೂನ್ಯ * output, uint64_t ಸ್ಥಾನಾಂತರ, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    ಆದರೆ (reader.ಸ್ಥಾನಾಂತರ != ಸ್ಥಾನಾಂತರ) {
        ಆದರೆ (ಸ್ಥಾನಾಂತರ > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(ಸ್ಥಾನಾಂತರ), SEEK_SET) != 0) {
            ಹಿಂತಿರುಗಿ 0;
        }

        reader.ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
    }

    ಸ್ಥಿರಾಂಕ size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.ಸ್ಥಾನಾಂತರ += nread;
    ಹಿಂತಿರುಗಿ nread;
}

ರಚನೆ gguf_context * gguf_init_from_file_ptr(FILE * file, ರಚನೆ gguf_init_params params) {
    ಆದರೆ (!file) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ಥಿರಾಂಕ int64_t cur = gguf_ftell(file);
    ಆದರೆ (cur < 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.ಸ್ಥಾನಾಂತರ, gguf_reader::file_remain(file));
    ಹಿಂತಿರುಗಿ gguf_init_from_reader(gr, params);
}

ರಚನೆ gguf_ಬಫರ್_reader {
    ಸ್ಥಿರಾಂಕ uint8_t * ದತ್ತಾಂಶ;
    size_t          size;
};

ಸ್ಥಿರ size_t gguf_ಬಫರ್_reader_callback(ಶೂನ್ಯ * userdata, ಶೂನ್ಯ * output, uint64_t ಸ್ಥಾನಾಂತರ, size_t len) {
    GGML_ASSERT(len > 0);

    ಸ್ಥಿರಾಂಕ gguf_ಬಫರ್_reader & reader = *static_cast<gguf_ಬಫರ್_reader *>(userdata);

    ಆದರೆ (ಸ್ಥಾನಾಂತರ > reader.size || len > reader.size - ಸ್ಥಾನಾಂತರ) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ size_t ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ = static_cast<size_t>(ಸ್ಥಾನಾಂತರ);
    ಸ್ಥಿರಾಂಕ size_t nread = std::min(len, reader.size - ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ);
    ಸ್ಮರಣೆ_ನಕಲು(static_cast<uint8_t *>(output), reader.ದತ್ತಾಂಶ + ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ, nread);
    ಹಿಂತಿರುಗಿ nread;
}

ರಚನೆ gguf_context * gguf_init_from_ಬಫರ್(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size, ರಚನೆ gguf_init_params params) {
    ಆದರೆ (ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ || size == 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    gguf_ಬಫರ್_reader reader = {
        /*.data = */ static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ),
        /*.size = */ size,
    };
    ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_reader gr(gguf_ಬಫರ್_reader_callback, &reader, SIZE_MAX, 0, size);
    ಹಿಂತಿರುಗಿ gguf_init_from_reader(gr, params);
}

ರಚನೆ gguf_context * gguf_init_from_file(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * fname, ರಚನೆ gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    ಆದರೆ (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ರಚನೆ gguf_context * ಫಲಿತಾಂಶ = gguf_init_from_file_ptr(file, params);
    fclose(file);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ gguf_free(ರಚನೆ gguf_context * ctx) {
    ಆದರೆ (ctx == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಅಳಿಸು ctx;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gguf_type_name(ಎಣಿಕೆ gguf_type type) {
    ಸ್ವಯಂಚಾಲಿತ it = GGUF_TYPE_NAME.find(type);
    ಹಿಂತಿರುಗಿ it == GGUF_TYPE_NAME.end() ? ಶೂನ್ಯಸೂಚಕ : it->second;
}

uint32_t gguf_get_ಆವೃತ್ತಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->ಆವೃತ್ತಿ_2;
}

size_t gguf_get_alignment(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->alignment;
}

size_t gguf_get_ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->ಸ್ಥಾನಾಂತರ;
}

int64_t gguf_get_n_kv(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->kv.size();
}

int64_t gguf_ಕಂಡುಹಿಡಿ_ಕೀಲಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ಸ್ಥಿರಾಂಕ int64_t n_kv = gguf_get_n_kv(ctx);

    ಗಾಗಿ (int64_t i = 0; i < n_kv; ++i) {
        ಆದರೆ (strcmp(ಕೀಲಿ, gguf_get_ಕೀಲಿ(ctx, i)) == 0) {
            keyfound = i;
            ನಿಲ್ಲಿಸು;
        }
    }

    ಹಿಂತಿರುಗಿ keyfound;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gguf_get_ಕೀಲಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_ಕೀಲಿ().c_str();
}

ಎಣಿಕೆ gguf_type gguf_get_kv_type(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[ಕೀಲಿ_id].get_type();
}

ಎಣಿಕೆ gguf_type gguf_get_arr_type(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].is_array);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_type();
}

ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * gguf_get_arr_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_type() != GGUF_TYPE_STRING);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].ದತ್ತಾಂಶ.ದತ್ತಾಂಶ();
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gguf_get_arr_str(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id, size_t i) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_type() == GGUF_TYPE_STRING);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].ದತ್ತಾಂಶ_string[i].c_str();
}

size_t gguf_get_arr_n(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));

    ಆದರೆ (ctx->kv[ಕೀಲಿ_id].type == GGUF_TYPE_STRING) {
        ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].ದತ್ತಾಂಶ_string.size();
    }

    ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = gguf_type_ಗಾತ್ರ(ctx->kv[ಕೀಲಿ_id].type);
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].ದತ್ತಾಂಶ.size() % type_ಗಾತ್ರ == 0);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].ದತ್ತಾಂಶ.size() / type_ಗಾತ್ರ;
}

uint8_t gguf_get_val_u8(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<int32_t>();
}

ದಶಮಾಂಶ gguf_get_val_f32(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<ದಶಮಾಂಶ>();
}

uint64_t gguf_get_val_u64(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<int64_t>();
}

ದ್ವಿಗುಣ gguf_get_val_f64(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<ದ್ವಿಗುಣ>();
}

ತಾರ್ಕಿಕ gguf_get_val_bool(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<ತಾರ್ಕಿಕ>();
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gguf_get_val_str(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].get_val<std::string>().c_str();
}

ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * gguf_get_val_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_type() != GGUF_TYPE_STRING);
    ಹಿಂತಿರುಗಿ ctx->kv[ಕೀಲಿ_id].ದತ್ತಾಂಶ.ದತ್ತಾಂಶ();
}

int64_t gguf_get_n_tensors(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx) {
    ಹಿಂತಿರುಗಿ ctx->info.size();
}

int64_t gguf_ಕಂಡುಹಿಡಿ_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ಸ್ಥಿರಾಂಕ int64_t n_tensors = gguf_get_n_tensors(ctx);

    ಗಾಗಿ (int64_t i = 0; i < n_tensors; ++i) {
        ಆದರೆ (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            ನಿಲ್ಲಿಸು;
        }
    }

    ಹಿಂತಿರುಗಿ tensor_id;
}

size_t gguf_get_tensor_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಹಿಂತಿರುಗಿ ctx->info[tensor_id].ಸ್ಥಾನಾಂತರ;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gguf_get_tensor_name(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಹಿಂತಿರುಗಿ ctx->info[tensor_id].t.name;
}

ಸ್ಥಿರಾಂಕ int64_t * gguf_get_tensor_ne(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಹಿಂತಿರುಗಿ ctx->info[tensor_id].t.ne;
}

ಎಣಿಕೆ ggml_type gguf_get_tensor_type(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಹಿಂತಿರುಗಿ ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಹಿಂತಿರುಗಿ ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_ಕೀಲಿ(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ) {
    ಸ್ಥಿರಾಂಕ int64_t ಕೀಲಿ_id = gguf_ಕಂಡುಹಿಡಿ_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ಆದರೆ (ಕೀಲಿ_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + ಕೀಲಿ_id);
    }
    ಹಿಂತಿರುಗಿ ಕೀಲಿ_id;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ gguf_check_reserved_ಕೀಲಿಗಳು(ಸ್ಥಿರಾಂಕ std::string & ಕೀಲಿ, ಸ್ಥಿರಾಂಕ T val) {
    ಆದರೆ (ಕೀಲಿ == GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT) {
        ಆದರೆ constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT " must be power of 2");
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ಬಳಸದ(val);
            GGML_ABORT(GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

ಶೂನ್ಯ gguf_ಗಣ_val_u8(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, uint8_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_i8(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, int8_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_u16(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, uint16_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_i16(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, int16_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_u32(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, uint32_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_i32(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, int32_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_f32(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, ದಶಮಾಂಶ val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_u64(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, uint64_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_i64(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, int64_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_f64(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, ದ್ವಿಗುಣ val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_bool(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, ತಾರ್ಕಿಕ val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_ಗಣ_val_str(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, std::string(val));
}

ಶೂನ್ಯ gguf_ಗಣ_arr_ದತ್ತಾಂಶ(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, ಎಣಿಕೆ gguf_type type, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t n) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, ದತ್ತಾಂಶ);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);

    ಸ್ಥಿರಾಂಕ size_t nbytes = n*gguf_type_ಗಾತ್ರ(type);
    std::vector<int8_t> tmp(nbytes);
    ಆದರೆ (!tmp.ಖಾಲಿಗಳು()) {
        ಸ್ಮರಣೆ_ನಕಲು(tmp.ದತ್ತಾಂಶ(), ದತ್ತಾಂಶ, nbytes);
    }
    ctx->kv.emplace_back(ಕೀಲಿ, tmp);
    ctx->kv.back().cast(type);
}

ಶೂನ್ಯ gguf_ಗಣ_arr_str(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ ** ದತ್ತಾಂಶ, size_t n) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, ದತ್ತಾಂಶ);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);

    std::vector<std::string> tmp(n);
    ಗಾಗಿ (size_t i = 0; i < n; ++i) {
        tmp[i] = ದತ್ತಾಂಶ[i];
    }
    ctx->kv.emplace_back(ಕೀಲಿ, tmp);
}

// set or add KV pairs from another context
ಶೂನ್ಯ gguf_ಗಣ_kv(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * src) {
    ಸ್ಥಿರಾಂಕ int64_t n_kv = gguf_get_n_kv(src);
    ಗಾಗಿ (int64_t i = 0; i < n_kv; ++i) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_kv & kv = src->kv[i];

        ಆದರೆ (!kv.is_array) {
            ಆಯ್ಕೆ (kv.get_type()) {
                ಸ್ಥಿತಿ GGUF_TYPE_UINT8:   gguf_ಗಣ_val_u8  (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint8_t>());             ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT8:    gguf_ಗಣ_val_i8  (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int8_t>());              ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_UINT16:  gguf_ಗಣ_val_u16 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint16_t>());            ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT16:   gguf_ಗಣ_val_i16 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int16_t>());             ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_UINT32:  gguf_ಗಣ_val_u32 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint32_t>());            ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT32:   gguf_ಗಣ_val_i32 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int32_t>());             ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_FLOAT32: gguf_ಗಣ_val_f32 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<ದಶಮಾಂಶ>());               ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_UINT64:  gguf_ಗಣ_val_u64 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint64_t>());            ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_INT64:   gguf_ಗಣ_val_i64 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int64_t>());             ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_FLOAT64: gguf_ಗಣ_val_f64 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<ದ್ವಿಗುಣ>());              ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_BOOL:    gguf_ಗಣ_val_bool(ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<ತಾರ್ಕಿಕ>());                ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_STRING:  gguf_ಗಣ_val_str (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<std::string>().c_str()); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGUF_TYPE_ARRAY:
                ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("invalid type");
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t ne = kv.get_ne();

        ಆಯ್ಕೆ (kv.get_type()) {
            ಸ್ಥಿತಿ GGUF_TYPE_UINT8:
            ಸ್ಥಿತಿ GGUF_TYPE_INT8:
            ಸ್ಥಿತಿ GGUF_TYPE_UINT16:
            ಸ್ಥಿತಿ GGUF_TYPE_INT16:
            ಸ್ಥಿತಿ GGUF_TYPE_UINT32:
            ಸ್ಥಿತಿ GGUF_TYPE_INT32:
            ಸ್ಥಿತಿ GGUF_TYPE_FLOAT32:
            ಸ್ಥಿತಿ GGUF_TYPE_UINT64:
            ಸ್ಥಿತಿ GGUF_TYPE_INT64:
            ಸ್ಥಿತಿ GGUF_TYPE_FLOAT64:
            ಸ್ಥಿತಿ GGUF_TYPE_BOOL: {
                gguf_ಗಣ_arr_ದತ್ತಾಂಶ(ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_type(), kv.ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), ne);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGUF_TYPE_STRING: {
                std::vector<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *> tmp(ne);
                ಗಾಗಿ (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.ದತ್ತಾಂಶ_string[j].c_str();
                }
                gguf_ಗಣ_arr_str(ctx, kv.get_ಕೀಲಿ().c_str(), tmp.ದತ್ತಾಂಶ(), ne);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGUF_TYPE_ARRAY:
            ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("invalid type");
        }
    }
}

ಶೂನ್ಯ gguf_ಕೂಡಿಸು_tensor(
             ರಚನೆ gguf_context * ctx,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    ಆದರೆ (gguf_ಕಂಡುಹಿಡಿ_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    ರಚನೆ gguf_tensor_info ti;
    ti.t = *tensor;
    ti.ಸ್ಥಾನಾಂತರ = ctx->info.ಖಾಲಿಗಳು() ? 0 :
        ctx->info.back().ಸ್ಥಾನಾಂತರ + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

ಶೂನ್ಯ gguf_ಗಣ_tensor_type(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ಎಣಿಕೆ ggml_type type) {
    ಸ್ಥಿರಾಂಕ int64_t tensor_id = gguf_ಕಂಡುಹಿಡಿ_tensor(ctx, name);
    ಆದರೆ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    ರಚನೆ ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ಸ್ಥಿರಾಂಕ size_t  type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(type);
    ಸ್ಥಿರಾಂಕ int64_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_ಗಾತ್ರ == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_ಗಾತ್ರ;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_ಗಾತ್ರ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ಸ್ಥಿರಾಂಕ int64_t n_tensors = gguf_get_n_tensors(ctx);
    ಗಾಗಿ (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].ಸ್ಥಾನಾಂತರ = ctx->info[i - 1].ಸ್ಥಾನಾಂತರ + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

ಶೂನ್ಯ gguf_ಗಣ_tensor_ದತ್ತಾಂಶ(ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ int64_t tensor_id = gguf_ಕಂಡುಹಿಡಿ_tensor(ctx, name);
    ಆದರೆ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.ದತ್ತಾಂಶ = (ಶೂನ್ಯ *)(uintptr_t)ದತ್ತಾಂಶ; // double cast suppresses warning about casting away const
}

ರಚನೆ gguf_writer_ತಳ {
    size_t written_ಬೈಟುಗಳು {0u};

    ವಾಸ್ತವಿಕ ~gguf_writer_ತಳ(ಶೂನ್ಯ) = ಪೂರ್ವನಿಯೋಜಿತ;

    // we bet on devirtualization
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ ಲಿಖ(int8_t val) = 0;
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ std::vector<int8_t> & val) = 0;
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ write_tensor_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_tensor_info & info, size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ, size_t alignment) = 0;

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ T & val) {
        ಗಾಗಿ (size_t i = 0; i < ಗಾತ್ರ(val); ++i) {
            ಲಿಖ(reinterpret_cast<ಸ್ಥಿರಾಂಕ int8_t *>(&val)[i]);
        }
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ & val) {
        ಸ್ಥಿರಾಂಕ int8_t val8 = val ? 1 : 0;
        ಲಿಖ(val8);
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ std::string & val) {
        {
            ಸ್ಥಿರಾಂಕ uint64_t n = val.ಉದ್ದಳತೆ();
            ಲಿಖ(n);
        }
        ಗಾಗಿ (size_t i = 0; i < val.ಉದ್ದಳತೆ(); ++i) {
            ಲಿಖ((val.ದತ್ತಾಂಶ())[i]);
        }
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * val) {
        ಲಿಖ(std::string(val));
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type & val) {
        ಲಿಖ(int32_t(val));
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ gguf_type & val) {
        ಲಿಖ(int32_t(val));
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_kv & kv) {
        ಸ್ಥಿರಾಂಕ uint64_t ne = kv.get_ne();

        ಲಿಖ(kv.get_ಕೀಲಿ());

        ಆದರೆ (kv.is_array) {
            ಲಿಖ(GGUF_TYPE_ARRAY);
            ಲಿಖ(kv.get_type());
            ಲಿಖ(ne);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಲಿಖ(kv.get_type());
        }

        ಆಯ್ಕೆ (kv.get_type()) {
            ಸ್ಥಿತಿ GGUF_TYPE_UINT8:
            ಸ್ಥಿತಿ GGUF_TYPE_INT8:
            ಸ್ಥಿತಿ GGUF_TYPE_UINT16:
            ಸ್ಥಿತಿ GGUF_TYPE_INT16:
            ಸ್ಥಿತಿ GGUF_TYPE_UINT32:
            ಸ್ಥಿತಿ GGUF_TYPE_INT32:
            ಸ್ಥಿತಿ GGUF_TYPE_FLOAT32:
            ಸ್ಥಿತಿ GGUF_TYPE_UINT64:
            ಸ್ಥಿತಿ GGUF_TYPE_INT64:
            ಸ್ಥಿತಿ GGUF_TYPE_FLOAT64: {
                ಲಿಖ(kv.ದತ್ತಾಂಶ);
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGUF_TYPE_BOOL: {
                ಗಾಗಿ (size_t i = 0; i < ne; ++i) {
                    ಲಿಖ(kv.get_val<ತಾರ್ಕಿಕ>(i));
                }
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGUF_TYPE_STRING: {
                ಗಾಗಿ (size_t i = 0; i < ne; ++i) {
                    ಲಿಖ(kv.get_val<std::string>(i));
                }
            } ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGUF_TYPE_ARRAY:
            ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("invalid type");
        }
    }

    ಶೂನ್ಯ write_tensor_meta(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_tensor_info & info) {
        ಲಿಖ(info.t.name);

        ಸ್ಥಿರಾಂಕ uint32_t n_dims = ggml_n_dims(&info.t);
        ಲಿಖ(n_dims);

        ಗಾಗಿ (uint32_t j = 0; j < n_dims; ++j) {
            ಲಿಖ(info.t.ne[j]);
        }
        ಲಿಖ(info.t.type);
        ಲಿಖ(info.ಸ್ಥಾನಾಂತರ);
    }

    ಶೂನ್ಯ pad(ಸ್ಥಿರಾಂಕ size_t alignment) {
        ಆಗ (written_ಬೈಟುಗಳು % alignment != 0) {
            ಸ್ಥಿರಾಂಕ int8_t zero = 0;
            ಲಿಖ(zero);
        }
    }
};

// vector buffer based writer
ರಚನೆ gguf_writer_buf final : ಸಾರ್ವಜನಿಕ gguf_writer_ತಳ {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    ಬಳಸಿ gguf_writer_ತಳ::ಲಿಖ;

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ int8_t val) override {
        buf.push_back(val);
        written_ಬೈಟುಗಳು++;
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_ಬೈಟುಗಳು += val.size();
    }

    ಶೂನ್ಯ write_tensor_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_tensor_info & info, ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ size_t alignment) override {
        GGML_ASSERT(buf.size() - ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ == info.ಸ್ಥಾನಾಂತರ);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = buf.size();
        ಸ್ಥಿರಾಂಕ size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(ಸ್ಥಾನಾಂತರ + nbytes);
        ಆದರೆ (info.t.ಬಫರ್) {
            ggml_backend_tensor_get(&info.t, buf.ದತ್ತಾಂಶ() + ಸ್ಥಾನಾಂತರ, 0, nbytes);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ASSERT(info.t.ದತ್ತಾಂಶ);
            ಸ್ಮರಣೆ_ನಕಲು(buf.ದತ್ತಾಂಶ() + ಸ್ಥಾನಾಂತರ, info.t.ದತ್ತಾಂಶ, nbytes);
        }
        written_ಬೈಟುಗಳು += nbytes;

        pad(alignment);
    }
};

// file based writer
ರಚನೆ gguf_writer_file final : ಸಾರ್ವಜನಿಕ gguf_writer_ತಳ {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    ಬಳಸಿ gguf_writer_ತಳ::ಲಿಖ;

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ int8_t val) override {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ real_val = static_cast<uint8_t>(val);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ret = fputc(real_val, file);
        written_ಬೈಟುಗಳು++;
        ಆದರೆ (ret != real_val) {
            ಎಸೆ std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((ಪೂರ್ಣಾಂಕ)real_val) + "'");
        }
    }

    ಶೂನ್ಯ ಲಿಖ(ಸ್ಥಿರಾಂಕ std::vector<int8_t> & val) override {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ret = fwrite(val.ದತ್ತಾಂಶ(), 1, val.size(), file);
        written_ಬೈಟುಗಳು += val.size();
        ಆದರೆ (ret != val.size()) {
            ಎಸೆ std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    ಶೂನ್ಯ write_tensor_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_tensor_info & info, ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ size_t alignment) override {
        GGML_ASSERT(written_ಬೈಟುಗಳು - ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ == info.ಸ್ಥಾನಾಂತರ);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ಸ್ಥಿರಾಂಕ size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        ಆದರೆ (info.t.ಬಫರ್) {
            ggml_backend_tensor_get(&info.t, buf.ದತ್ತಾಂಶ(), 0, nbytes);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ASSERT(info.t.ದತ್ತಾಂಶ);
            ಸ್ಮರಣೆ_ನಕಲು(buf.ದತ್ತಾಂಶ(), info.t.ದತ್ತಾಂಶ, nbytes);
        }
        ಲಿಖ(buf);

        pad(alignment);
    }
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು writer_t>
ಸ್ಥಿರ ಶೂನ್ಯ gguf_write_ಹೊರಹೋಗು(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, writer_t & gw, ತಾರ್ಕಿಕ only_meta) {
    ಸ್ಥಿರಾಂಕ int64_t n_kv      = gguf_get_n_kv(ctx);
    ಸ್ಥಿರಾಂಕ int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.ಲಿಖ(GGUF_MAGIC[0]);
    gw.ಲಿಖ(GGUF_MAGIC[1]);
    gw.ಲಿಖ(GGUF_MAGIC[2]);
    gw.ಲಿಖ(GGUF_MAGIC[3]);
    gw.ಲಿಖ(ctx->ಆವೃತ್ತಿ_2);
    gw.ಲಿಖ(n_tensors);
    gw.ಲಿಖ(n_kv);

    // write key-value pairs
    ಗಾಗಿ (int64_t i = 0; i < n_kv; ++i) {
        gw.ಲಿಖ(ctx->kv[i]);
    }

    // write tensor info
    ಗಾಗಿ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    ಆದರೆ (only_meta) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ = gw.written_ಬೈಟುಗಳು;

    // write tensor data
    ಗಾಗಿ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_ದತ್ತಾಂಶ(ctx->info[i], ಸ್ಥಾನಾಂತರ_ದತ್ತಾಂಶ, ctx->alignment);
    }
}

ಶೂನ್ಯ gguf_write_ಗಮ್ಯ_buf(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, std::vector<int8_t> & buf, ತಾರ್ಕಿಕ only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_ಹೊರಹೋಗು(ctx, gw, only_meta);
}

ತಾರ್ಕಿಕ gguf_write_ಗಮ್ಯ_file_ptr(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, FILE * file, ತಾರ್ಕಿಕ only_meta) {
    GGML_ASSERT(file);

    ಪ್ರಯತ್ನಿಸು {
        gguf_writer_file gw(file);
        gguf_write_ಹೊರಹೋಗು(ctx, gw, only_meta);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ gguf_write_ಗಮ್ಯ_file(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * fname, ತಾರ್ಕಿಕ only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    ಆದರೆ (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ success = gguf_write_ಗಮ್ಯ_file_ptr(ctx, file, only_meta);
    ಆದರೆ (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    ಹಿಂತಿರುಗಿ success;
}

size_t gguf_get_meta_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_ಗಮ್ಯ_buf(ctx, buf, /*only_meta =*/ ನಿಜ);
    ಹಿಂತಿರುಗಿ buf.size();
}

ಶೂನ್ಯ gguf_get_meta_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    std::vector<int8_t> buf;
    gguf_write_ಗಮ್ಯ_buf(ctx, buf, /*only_meta =*/ ನಿಜ);
    ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, buf.ದತ್ತಾಂಶ(), buf.size());
}
