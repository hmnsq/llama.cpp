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

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#ಅನ್ಯಥಾ
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
ಸಂರಚನಾ type_to_gguf_type;

template <>
ಸಂರಚನಾ type_to_gguf_type<uint8_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_UINT8;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<int8_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_INT8;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<uint16_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_UINT16;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<int16_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_INT16;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<uint32_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_UINT32;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<int32_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_INT32;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<ಭಿನ್ನ> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<ತಾರ್ಕಿಕ> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_BOOL;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<std::string> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_STRING;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<uint64_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_UINT64;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<int64_t> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_INT64;
};

template <>
ಸಂರಚನಾ type_to_gguf_type<ದ್ವಿಗುಣ> {
    ಸ್ಥಿರ constexpr ಗಣನಾ gguf_type value = GGUF_TYPE_FLOAT64;
};

ಸ್ಥಿರ ಧ್ರುವ std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   ಪರಿಮಾಣ(uint8_t)},
    {GGUF_TYPE_INT8,    ಪರಿಮಾಣ(int8_t)},
    {GGUF_TYPE_UINT16,  ಪರಿಮಾಣ(uint16_t)},
    {GGUF_TYPE_INT16,   ಪರಿಮಾಣ(int16_t)},
    {GGUF_TYPE_UINT32,  ಪರಿಮಾಣ(uint32_t)},
    {GGUF_TYPE_INT32,   ಪರಿಮಾಣ(int32_t)},
    {GGUF_TYPE_FLOAT32, ಪರಿಮಾಣ(ಭಿನ್ನ)},
    {GGUF_TYPE_BOOL,    ಪರಿಮಾಣ(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  ಪರಿಮಾಣ(uint64_t)},
    {GGUF_TYPE_INT64,   ಪರಿಮಾಣ(int64_t)},
    {GGUF_TYPE_FLOAT64, ಪರಿಮಾಣ(ದ್ವಿಗುಣ)},
};
static_assert(GGUF_TYPE_ಎಣಿಕೆ == 13, "GGUF_TYPE_COUNT != 13");

ಸ್ಥಿರ ಧ್ರುವ std::map<gguf_type, ಧ್ರುವ ವರ್ಣ *> GGUF_TYPE_NAME = {
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

size_t gguf_type_size(ಗಣನಾ gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    ಪ್ರತ್ಯಾಗಮ it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

ಸಂರಚನಾ gguf_kv {
    std::string ಕೀಲಿ;

    ತಾರ್ಕಿಕ is_array;
    ಗಣನಾ gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(ಧ್ರುವ std::string & ಕೀಲಿ, ಧ್ರುವ T value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ಅಸತ್ಯ), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!ಕೀಲಿ.empty());
        data.resize(ಪರಿಮಾಣ(T));
        memcpy(data.data(), &value, ಪರಿಮಾಣ(T));
    }

    template <typename T>
    gguf_kv(ಧ್ರುವ std::string & ಕೀಲಿ, ಧ್ರುವ std::vector<T> & value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ಸತ್ಯ), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!ಕೀಲಿ.empty());
        data.resize(value.size()*ಪರಿಮಾಣ(T));
        ಪ್ರತಿ (size_t i = 0; i < value.size(); ++i) {
            ಧ್ರುವ T tmp = value[i];
            memcpy(data.data() + i*ಪರಿಮಾಣ(T), &tmp, ಪರಿಮಾಣ(T));
        }
    }

    gguf_kv(ಧ್ರುವ std::string & ಕೀಲಿ, ಧ್ರುವ std::string & value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ಅಸತ್ಯ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ಕೀಲಿ.empty());
        data_string.push_back(value);
    }

    gguf_kv(ಧ್ರುವ std::string & ಕೀಲಿ, ಧ್ರುವ std::vector<std::string> & value)
            : ಕೀಲಿ(ಕೀಲಿ), is_array(ಸತ್ಯ), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!ಕೀಲಿ.empty());
        data_string = value;
    }

    ಧ್ರುವ std::string & get_ಕೀಲಿ() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ಕೀಲಿ;
    }

    ಧ್ರುವ ಗಣನಾ gguf_type & get_type() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type;
    }

    size_t get_ne() ಧ್ರುವ {
        ಯದಿ (type == GGUF_TYPE_STRING) {
            ಧ್ರುವ size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            ಪ್ರತ್ಯಾಗಮ ne;
        }
        ಧ್ರುವ size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        ಧ್ರುವ size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        ಪ್ರತ್ಯಾಗಮ ne;
    }

    template <typename T>
    ಧ್ರುವ T & get_val(ಧ್ರುವ size_t i = 0) ಧ್ರುವ {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        ಯದಿ constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            ಪ್ರತ್ಯಾಗಮ data_string[i];
        }
        ಧ್ರುವ size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        ಪ್ರತ್ಯಾಗಮ reinterpret_cast<ಧ್ರುವ T *>(data.data())[i];
    }

    ಶೂನ್ಯ cast(ಧ್ರುವ ಗಣನಾ gguf_type new_type) {
        ಧ್ರುವ size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

ಸಂರಚನಾ gguf_tensor_info {
    ಸಂರಚನಾ ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

ಸಂರಚನಾ gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<ಸಂರಚನಾ gguf_kv> kv;
    std::vector<ಸಂರಚನಾ gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    ಶೂನ್ಯ * data = nullptr;
};

ಸಂರಚನಾ gguf_reader {
    gguf_reader(
            gguf_reader_callback_t callback,
            ಶೂನ್ಯ * userdata,
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
    ಸ್ಥಿರ uint64_t file_remain(FILE * file) {
        ಧ್ರುವ int64_t cur = gguf_ftell(file);
        ಯದಿ (cur < 0) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }
        ಯದಿ (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ಪ್ರತ್ಯಾಗಮ 0;
        }
        ಧ್ರುವ int64_t end = gguf_ftell(file);
        ಯದಿ (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            ಪ್ರತ್ಯಾಗಮ 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        ಪ್ರತ್ಯಾಗಮ static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    ತಾರ್ಕಿಕ ಪಠ(T & dst) ಧ್ರುವ {
        ಧ್ರುವ size_t size = ಪರಿಮಾಣ(dst);
        ಯದಿ (size > nbytes_remain) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ read_raw(&dst, size) == size;
    }

    template <typename T>
    ತಾರ್ಕಿಕ ಪಠ(std::vector<T> & dst, ಧ್ರುವ size_t n) ಧ್ರುವ {
        ಯದಿ (n > GGUF_MAX_ARRAY_ELEMENTS) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            ಯದಿ (n > SIZE_MAX / ಪರಿಮಾಣ(uint64_t)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (nbytes_remain < n * ಪರಿಮಾಣ(uint64_t)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        } ಅನ್ಯಥಾ {
            ಯದಿ (n > SIZE_MAX / ಪರಿಮಾಣ(T)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (nbytes_remain < n * ಪರಿಮಾಣ(T)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        dst.resize(n);
        ಪ್ರತಿ (size_t i = 0; i < dst.size(); ++i) {
            ಯದಿ constexpr (std::is_same<T, ತಾರ್ಕಿಕ>::value) {
                ತಾರ್ಕಿಕ tmp;
                ಯದಿ (!ಪಠ(tmp)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                dst[i] = tmp;
            } ಅನ್ಯಥಾ {
                ಯದಿ (!ಪಠ(dst[i])) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ ಪಠ(ತಾರ್ಕಿಕ & dst) ಧ್ರುವ {
        int8_t tmp = -1;
        ಯದಿ (!ಪಠ(tmp)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        dst = tmp != 0;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ ಪಠ(ಗಣನಾ ggml_type & dst) ಧ್ರುವ {
        int32_t tmp = -1;
        ಯದಿ (!ಪಠ(tmp)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        dst = ggml_type(tmp);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ ಪಠ(ಗಣನಾ gguf_type & dst) ಧ್ರುವ {
        int32_t tmp = -1;
        ಯದಿ (!ಪಠ(tmp)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        dst = gguf_type(tmp);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ ಪಠ(std::string & dst) ಧ್ರುವ {
        uint64_t size = 0;
        ಯದಿ (!ಪಠ(size)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        dst.resize(static_cast<size_t>(size));
        ಪ್ರತ್ಯಾಗಮ read_raw(dst.data(), static_cast<size_t>(size)) == size;
    }

    ತಾರ್ಕಿಕ ಪಠ(ಶೂನ್ಯ * dst, ಧ್ರುವ size_t size) ಧ್ರುವ {
        ಯದಿ (size > nbytes_remain) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ read_raw(dst, size) == size;
    }

    uint64_t tell() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ data_offset;
    }

    // position in the file where the GGUF data starts, alignment is relative to it, not to the file
    uint64_t start() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ start_offset;
    }

    ತಾರ್ಕಿಕ seek(uint64_t absolute_offset) ಧ್ರುವ {
        ಧ್ರುವ uint64_t end_offset = uint64_t(data_offset) + nbytes_remain;
        ಯದಿ (absolute_offset > end_offset) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        data_offset = absolute_offset;
        nbytes_remain = end_offset - absolute_offset;

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

ಗುಪ್ತ:
    size_t read_raw(ಶೂನ್ಯ * dst, size_t size) ಧ್ರುವ {
        ಯದಿ (callback == nullptr || size == 0) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        uint8_t * data = static_cast<uint8_t *>(dst);
        size_t total_nread = 0;
        ತಾರ್ಕಿಕ reached_eof = ಅಸತ್ಯ;

        ಯಾವತ್ (total_nread < size) {
            ಧ್ರುವ size_t chunk_size = std::min(max_chunk_read, size - total_nread);
            ಯದಿ (data_offset + total_nread < data_offset) {
                ಭಙ್ಗ;
            }
            ಧ್ರುವ size_t nread = callback(userdata, static_cast<ಶೂನ್ಯ *>(data + total_nread), data_offset + total_nread, chunk_size);
            total_nread += nread;
            ಯದಿ (nread != chunk_size) {
                reached_eof = ಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }

        data_offset += total_nread;
        GGML_ASSERT(total_nread <= nbytes_remain);
        nbytes_remain -= total_nread;

        ಯದಿ (reached_eof) {
            nbytes_remain = 0;
        }

        ಪ್ರತ್ಯಾಗಮ total_nread;
    }

    gguf_reader_callback_t callback = nullptr;
    ಶೂನ್ಯ * userdata = nullptr;
    size_t max_chunk_read = 0;
    uint64_t start_offset = 0;
    mutable uint64_t data_offset = 0;
    mutable uint64_t nbytes_remain = 0;
};

ಸಂರಚನಾ gguf_context * gguf_init_empty(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ ನೂತನ gguf_context;
}

template<typename T>
ತಾರ್ಕಿಕ gguf_read_emplace_helper(ಧ್ರುವ ಸಂರಚನಾ gguf_reader & gr, std::vector<ಸಂರಚನಾ gguf_kv> & kv, ಧ್ರುವ std::string & ಕೀಲಿ, ಧ್ರುವ ತಾರ್ಕಿಕ is_array, ಧ್ರುವ size_t n) {
    ಯದಿ (is_array) {
        std::vector<T> value;
        ಯತ್ನ {
            ಯದಿ (!gr.ಪಠ(value, n)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        } ಗೃಹಾಣ (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, ಕೀಲಿ.c_str());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        } ಗೃಹಾಣ (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, ಕೀಲಿ.c_str());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        kv.emplace_back(ಕೀಲಿ, value);
    } ಅನ್ಯಥಾ {
        T value;
        ಯದಿ (!gr.ಪಠ(value)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        kv.emplace_back(ಕೀಲಿ, value);
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಸಂರಚನಾ gguf_context * gguf_init_from_reader(ಧ್ರುವ ಸಂರಚನಾ gguf_reader & gr, ಸಂರಚನಾ gguf_init_params params) {
    ಸಂರಚನಾ gguf_context * ctx = ನೂತನ gguf_context;

    ತಾರ್ಕಿಕ ok = ಸತ್ಯ;

    // file magic
    {
        std::vector<ವರ್ಣ> magic;
        ok = ok && gr.ಪಠ(magic, 4);

        ಯದಿ (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        ಪ್ರತಿ (uint32_t i = 0; i < magic.size(); i++) {
            ಯದಿ (magic[i] != GGUF_MAGIC[i]) {
                ವರ್ಣ c0 = isprint(magic[0]) ? magic[0] : '?';
                ವರ್ಣ c1 = isprint(magic[1]) ? magic[1] : '?';
                ವರ್ಣ c2 = isprint(magic[2]) ? magic[2] : '?';
                ವರ್ಣ c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    ಯದಿ (ok && gr.ಪಠ(ctx->version)) {
        ಯದಿ (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = ಅಸತ್ಯ;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        ಯದಿ (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = ಅಸತ್ಯ;
        }

        ಯದಿ (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = ಅಸತ್ಯ;
        }
        ಯದಿ (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ok = ಅಸತ್ಯ;
    }

    ಯದಿ (ok && gr.ಪಠ(n_tensors)) {
        static_assert(ಪರಿಮಾಣ(size_t) <= 8 && ಪರಿಮಾಣ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ಯದಿ (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/ಪರಿಮಾಣ(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/ಪರಿಮಾಣ(gguf_tensor_info));
            ok = ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ok = ಅಸತ್ಯ;
    }

    ಯದಿ (ok && gr.ಪಠ(n_kv)) {
        static_assert(ಪರಿಮಾಣ(size_t) <= 8 && ಪರಿಮಾಣ(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        ಯದಿ (n_kv < 0 || n_kv > int64_t(SIZE_MAX/ಪರಿಮಾಣ(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/ಪರಿಮಾಣ(gguf_kv));
            ok = ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ok = ಅಸತ್ಯ;
    }

    ಯದಿ (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    // KV pairs
    {
        std::unordered_set<std::string> seen_ಕೀಲಿಗಳು;

        ಪ್ರತಿ (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string ಕೀಲಿ;
            gguf_type   type     = gguf_type(-1);
            ತಾರ್ಕಿಕ        is_array = ಅಸತ್ಯ;
            uint64_t    n        = 1;

            ಯತ್ನ {
                ok = ok && gr.ಪಠ(ಕೀಲಿ);
            } ಗೃಹಾಣ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = ಅಸತ್ಯ;
            } ಗೃಹಾಣ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = ಅಸತ್ಯ;
            }
            ಯದಿ (ok && ಕೀಲಿ.empty()) {
                GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, i);
                ok = ಅಸತ್ಯ;
            }
            ಯದಿ (ok && !seen_ಕೀಲಿಗಳು.insert(ಕೀಲಿ).second) {
                GGML_LOG_ERROR("%s: duplicate key '%s' for KV pair %" PRIi64 "\n", __func__, ಕೀಲಿ.c_str(), i);
                ok = ಅಸತ್ಯ;
            }
            ಯದಿ (!ok) {
                ಭಙ್ಗ;
            }

            ok = ok && gr.ಪಠ(type);
            ಯದಿ (type == GGUF_TYPE_ARRAY) {
                is_array = ಸತ್ಯ;
                ok = ok && gr.ಪಠ(type);
                ok = ok && gr.ಪಠ(n);
            }
            ಯದಿ (!ok) {
                ಭಙ್ಗ;
            }

            ವಿಕಲ್ಪ (type) {
                ಪಕ್ಷ GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<ಭಿನ್ನ>      (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<ತಾರ್ಕಿಕ>       (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<ದ್ವಿಗುಣ>     (gr, ctx->kv, ಕೀಲಿ, is_array, n); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_ARRAY:
                ಸ್ವಭಾವ:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, ಕೀಲಿ.c_str(), type);
                        ok = ಅಸತ್ಯ;
                    } ಭಙ್ಗ;
            }
        }

        ಯದಿ (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ alignment_idx = gguf_find_ಕೀಲಿ(ctx, GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT);
        ಯದಿ (alignment_idx != -1 && gguf_get_kv_type(ctx, alignment_idx) != GGUF_TYPE_UINT32) {
            GGML_LOG_ERROR("%s: key '%s' must be of type %s but is %s\n",
                __func__, GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT, gguf_type_name(GGUF_TYPE_UINT32),
                gguf_type_name(gguf_get_kv_type(ctx, alignment_idx)));
            gguf_free(ctx);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        ಯದಿ (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }

    // read the tensor info
    std::unordered_set<std::string> seen_tensor_names;

    ಪ್ರತಿ (int64_t i = 0; ok && i < n_tensors; ++i) {
        ಸಂರಚನಾ gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            ಯತ್ನ {
                ok = ok && gr.ಪಠ(name);
            } ಗೃಹಾಣ (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ಅಸತ್ಯ;
            } ಗೃಹಾಣ (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = ಅಸತ್ಯ;
            }
            ಯದಿ (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            ಯದಿ (ok && !seen_tensor_names.insert(name).second) {
                GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensor %" PRIi64 "\n", __func__, info.t.name, i);
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (!ok) {
            ಭಙ್ಗ;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.ಪಠ(n_dims);
            ಯದಿ (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
            ಪ್ರತಿ (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                ಯದಿ (j < n_dims) {
                    ok = ok && gr.ಪಠ(info.t.ne[j]);
                }

                // check that all ne are non-negative
                ಯದಿ (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }

            // check that the total number of elements is representable
            // (a zero-element tensor is trivially representable)
            ಯದಿ (ok) {
                int64_t current_nelements = info.t.ne[0];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                    ಯದಿ (info.t.ne[i] > 1) { // 1 or 0 won't cause an overflow
                        ಯದಿ (INT64_MAX / info.t.ne[i] < current_nelements) {
                             ok = ಅಸತ್ಯ;
                             ಭಙ್ಗ;
                         }
                     }
                     current_nelements *= info.t.ne[i];
                 }
                 ಯದಿ (!ok) {
                     GGML_LOG_ERROR("%s: incorrect shape for a tensor '%s' may cause overflow errors: "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= INT64_MAX (%" PRIi64 ")\n",
                         __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                     ಭಙ್ಗ;
                 }
            }
        }
        ಯದಿ (!ok) {
            ಭಙ್ಗ;
        }

        // tensor type
        {
            ok = ok && gr.ಪಠ(info.t.type);

            // check that tensor type is within defined range
            ಯದಿ (info.t.type < 0 || info.t.type >= GGML_TYPE_ಎಣಿಕೆ) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_ಎಣಿಕೆ);
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
            ಧ್ರುವ size_t  type_size = ggml_type_size(info.t.type);
            ಧ್ರುವ int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            ಯದಿ (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (ಪೂರ್ಣಾಙ್ಕ) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }

            // check that the size of the tensor in bytes is representable
            ಯದಿ (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        ಯದಿ (!ok) {
            ಭಙ್ಗ;
        }

        // tensor data offset within buffer
        ok = ok && gr.ಪಠ(info.offset);

        ctx->info.push_back(info);
    }

    ಯದಿ (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    ಯದಿ (n_tensors > 0 && !gr.seek(gr.start() + GGML_PAD(gr.tell() - gr.start(), ctx->alignment))) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gr.tell();

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        ಪ್ರತಿ (size_t i = 0; i < ctx->info.size(); ++i) {
            ಧ್ರುವ gguf_tensor_info & ti = ctx->info[i];
            ಯದಿ (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ಧ್ರುವ size_t nbytes = ggml_nbytes(&ti.t);
            // GGML_PAD adds (alignment - 1) to nbytes. when nbytes is near
            // SIZE_MAX this wraps to 0 and makes the check below dead.
            ಯದಿ (nbytes > SIZE_MAX - (ctx->alignment - 1)) {
                GGML_LOG_ERROR("%s: tensor '%s' size %zu overflows after padding (alignment %zu)\n",
                    __func__, ti.t.name, nbytes, ctx->alignment);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ಧ್ರುವ size_t padded_size = GGML_PAD(nbytes, ctx->alignment);
            ಯದಿ (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    ಯದಿ (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        ಯದಿ (params.no_alloc) {
            ಯದಿ (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }

            ಧ್ರುವ size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } ಅನ್ಯಥಾ {
            ಯದಿ ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }

            ಧ್ರುವ size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            ಯದಿ (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        ಸಂರಚನಾ ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        ಯದಿ (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        ಸಂರಚನಾ ggml_context * ctx_data = *params.ctx;

        ಸಂರಚನಾ ggml_tensor * data = nullptr;

        ಯದಿ (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            ಯದಿ (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.ಪಠ(data->data, ctx->size);

            ಯದಿ (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, ಸತ್ಯ);

        // create the tensors
        ಪ್ರತಿ (size_t i = 0; i < ctx->info.size(); ++i) {
            ಧ್ರುವ ಸಂರಚನಾ gguf_tensor_info & info = ctx->info[i];

            ಸಂರಚನಾ ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            ಯದಿ (!ok) {
                ಭಙ್ಗ;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            ಯದಿ (!params.no_alloc) {
                cur->data = (ವರ್ಣ *) data->data + info.offset;
            }
        }

        ಯದಿ (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    ಪ್ರತ್ಯಾಗಮ ctx;
}

ಸಂರಚನಾ gguf_context * gguf_init_from_callback(gguf_reader_callback_t callback, ಶೂನ್ಯ * userdata, size_t max_chunk_read, uint64_t max_expected_size, ಸಂರಚನಾ gguf_init_params params) {
    ಯದಿ (callback == nullptr) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಧ್ರುವ ಸಂರಚನಾ gguf_reader gr(callback, userdata, max_chunk_read == 0 ? SIZE_MAX : max_chunk_read, 0, max_expected_size);
    ಪ್ರತ್ಯಾಗಮ gguf_init_from_reader(gr, params);
}

ಸಂರಚನಾ gguf_file_reader {
    FILE * file;
    uint64_t offset;
};

ಸ್ಥಿರ size_t gguf_file_reader_callback(ಶೂನ್ಯ * userdata, ಶೂನ್ಯ * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    gguf_file_reader & reader = *static_cast<gguf_file_reader *>(userdata);

    ಯದಿ (reader.offset != offset) {
        ಯದಿ (offset > INT64_MAX || gguf_fseek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        reader.offset = offset;
    }

    ಧ್ರುವ size_t nread = fread(static_cast<uint8_t *>(output), 1, len, reader.file);
    reader.offset += nread;
    ಪ್ರತ್ಯಾಗಮ nread;
}

ಸಂರಚನಾ gguf_context * gguf_init_from_file_ptr(FILE * file, ಸಂರಚನಾ gguf_init_params params) {
    ಯದಿ (!file) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಧ್ರುವ int64_t cur = gguf_ftell(file);
    ಯದಿ (cur < 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    gguf_file_reader reader = {
        /*.file   = */ file,
        /*.offset = */ static_cast<uint64_t>(cur),
    };
    ಧ್ರುವ ಸಂರಚನಾ gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
    ಪ್ರತ್ಯಾಗಮ gguf_init_from_reader(gr, params);
}

ಸಂರಚನಾ gguf_buffer_reader {
    ಧ್ರುವ uint8_t * data;
    size_t          size;
};

ಸ್ಥಿರ size_t gguf_buffer_reader_callback(ಶೂನ್ಯ * userdata, ಶೂನ್ಯ * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    ಧ್ರುವ gguf_buffer_reader & reader = *static_cast<gguf_buffer_reader *>(userdata);

    ಯದಿ (offset > reader.size || len > reader.size - offset) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ size_t data_offset = static_cast<size_t>(offset);
    ಧ್ರುವ size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    ಪ್ರತ್ಯಾಗಮ nread;
}

ಸಂರಚನಾ gguf_context * gguf_init_from_buffer(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size, ಸಂರಚನಾ gguf_init_params params) {
    ಯದಿ (data == nullptr || size == 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    gguf_buffer_reader reader = {
        /*.data = */ static_cast<ಧ್ರುವ uint8_t *>(data),
        /*.size = */ size,
    };
    ಧ್ರುವ ಸಂರಚನಾ gguf_reader gr(gguf_buffer_reader_callback, &reader, SIZE_MAX, 0, size);
    ಪ್ರತ್ಯಾಗಮ gguf_init_from_reader(gr, params);
}

ಸಂರಚನಾ gguf_context * gguf_init_from_file(ಧ್ರುವ ವರ್ಣ * fname, ಸಂರಚನಾ gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    ಯದಿ (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಸಂರಚನಾ gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    ಪ್ರತ್ಯಾಗಮ result;
}

ಶೂನ್ಯ gguf_free(ಸಂರಚನಾ gguf_context * ctx) {
    ಯದಿ (ctx == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    delete ctx;
}

ಧ್ರುವ ವರ್ಣ * gguf_type_name(ಗಣನಾ gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    ಪ್ರತ್ಯಾಗಮ it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->version;
}

size_t gguf_get_alignment(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->alignment;
}

size_t gguf_get_data_offset(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->offset;
}

int64_t gguf_get_n_kv(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->kv.size();
}

int64_t gguf_find_ಕೀಲಿ(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ) {
    // return -1 if key not found
    int64_t keyfound = -1;

    ಧ್ರುವ int64_t n_kv = gguf_get_n_kv(ctx);

    ಪ್ರತಿ (int64_t i = 0; i < n_kv; ++i) {
        ಯದಿ (strcmp(ಕೀಲಿ, gguf_get_ಕೀಲಿ(ctx, i)) == 0) {
            keyfound = i;
            ಭಙ್ಗ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ keyfound;
}

ಧ್ರುವ ವರ್ಣ * gguf_get_ಕೀಲಿ(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_ಕೀಲಿ().c_str();
}

ಗಣನಾ gguf_type gguf_get_kv_type(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[ಕೀಲಿ_id].get_type();
}

ಗಣನಾ gguf_type gguf_get_arr_type(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].is_array);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_type();
}

ಧ್ರುವ ಶೂನ್ಯ * gguf_get_arr_data(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_type() != GGUF_TYPE_STRING);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].data.data();
}

ಧ್ರುವ ವರ್ಣ * gguf_get_arr_str(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id, size_t i) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_type() == GGUF_TYPE_STRING);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));

    ಯದಿ (ctx->kv[ಕೀಲಿ_id].type == GGUF_TYPE_STRING) {
        ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].data_string.size();
    }

    ಧ್ರುವ size_t type_size = gguf_type_size(ctx->kv[ಕೀಲಿ_id].type);
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].data.size() % type_size == 0);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<int32_t>();
}

ಭಿನ್ನ gguf_get_val_f32(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<ಭಿನ್ನ>();
}

uint64_t gguf_get_val_u64(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<int64_t>();
}

ದ್ವಿಗುಣ gguf_get_val_f64(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<ದ್ವಿಗುಣ>();
}

ತಾರ್ಕಿಕ gguf_get_val_bool(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<ತಾರ್ಕಿಕ>();
}

ಧ್ರುವ ವರ್ಣ * gguf_get_val_str(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].get_val<std::string>().c_str();
}

ಧ್ರುವ ಶೂನ್ಯ * gguf_get_val_data(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t ಕೀಲಿ_id) {
    GGML_ASSERT(ಕೀಲಿ_id >= 0 && ಕೀಲಿ_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[ಕೀಲಿ_id].get_type() != GGUF_TYPE_STRING);
    ಪ್ರತ್ಯಾಗಮ ctx->kv[ಕೀಲಿ_id].data.data();
}

int64_t gguf_get_n_tensors(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx) {
    ಪ್ರತ್ಯಾಗಮ ctx->info.size();
}

int64_t gguf_find_tensor(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    ಧ್ರುವ int64_t n_tensors = gguf_get_n_tensors(ctx);

    ಪ್ರತಿ (int64_t i = 0; i < n_tensors; ++i) {
        ಯದಿ (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            ಭಙ್ಗ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ tensor_id;
}

size_t gguf_get_tensor_offset(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಪ್ರತ್ಯಾಗಮ ctx->info[tensor_id].offset;
}

ಧ್ರುವ ವರ್ಣ * gguf_get_tensor_name(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಪ್ರತ್ಯಾಗಮ ctx->info[tensor_id].t.name;
}

ಧ್ರುವ int64_t * gguf_get_tensor_ne(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಪ್ರತ್ಯಾಗಮ ctx->info[tensor_id].t.ne;
}

ಗಣನಾ ggml_type gguf_get_tensor_type(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಪ್ರತ್ಯಾಗಮ ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_ಕೀಲಿ(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ) {
    ಧ್ರುವ int64_t ಕೀಲಿ_id = gguf_find_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ಯದಿ (ಕೀಲಿ_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + ಕೀಲಿ_id);
    }
    ಪ್ರತ್ಯಾಗಮ ಕೀಲಿ_id;
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ gguf_check_reserved_ಕೀಲಿಗಳು(ಧ್ರುವ std::string & ಕೀಲಿ, ಧ್ರುವ T val) {
    ಯದಿ (ಕೀಲಿ == GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT) {
        ಯದಿ constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT " must be power of 2");
        } ಅನ್ಯಥಾ {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_ಕೀಲಿ_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

ಶೂನ್ಯ gguf_set_val_u8(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, uint8_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_i8(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, int8_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_u16(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, uint16_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_i16(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, int16_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_u32(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, uint32_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_i32(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, int32_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_f32(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, ಭಿನ್ನ val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_u64(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, uint64_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_i64(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, int64_t val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_f64(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, ದ್ವಿಗುಣ val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_bool(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, ತಾರ್ಕಿಕ val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, val);
}

ಶೂನ್ಯ gguf_set_val_str(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, ಧ್ರುವ ವರ್ಣ * val) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, val);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);
    ctx->kv.emplace_back(ಕೀಲಿ, std::string(val));
}

ಶೂನ್ಯ gguf_set_arr_data(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, ಗಣನಾ gguf_type type, ಧ್ರುವ ಶೂನ್ಯ * data, size_t n) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, data);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);

    ಧ್ರುವ size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    ಯದಿ (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(ಕೀಲಿ, tmp);
    ctx->kv.back().cast(type);
}

ಶೂನ್ಯ gguf_set_arr_str(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ, ಧ್ರುವ ವರ್ಣ ** data, size_t n) {
    gguf_check_reserved_ಕೀಲಿಗಳು(ಕೀಲಿ, data);
    gguf_remove_ಕೀಲಿ(ctx, ಕೀಲಿ);

    std::vector<std::string> tmp(n);
    ಪ್ರತಿ (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(ಕೀಲಿ, tmp);
}

// set or add KV pairs from another context
ಶೂನ್ಯ gguf_set_kv(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ಸಂರಚನಾ gguf_context * src) {
    ಧ್ರುವ int64_t n_kv = gguf_get_n_kv(src);
    ಪ್ರತಿ (int64_t i = 0; i < n_kv; ++i) {
        ಧ್ರುವ ಸಂರಚನಾ gguf_kv & kv = src->kv[i];

        ಯದಿ (!kv.is_array) {
            ವಿಕಲ್ಪ (kv.get_type()) {
                ಪಕ್ಷ GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint8_t>());             ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int8_t>());              ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint16_t>());            ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int16_t>());             ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint32_t>());            ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int32_t>());             ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<ಭಿನ್ನ>());               ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<uint64_t>());            ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<int64_t>());             ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<ದ್ವಿಗುಣ>());              ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<ತಾರ್ಕಿಕ>());                ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_val<std::string>().c_str()); ಭಙ್ಗ;
                ಪಕ್ಷ GGUF_TYPE_ARRAY:
                ಸ್ವಭಾವ: GGML_ABORT("invalid type");
            }
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ size_t ne = kv.get_ne();

        ವಿಕಲ್ಪ (kv.get_type()) {
            ಪಕ್ಷ GGUF_TYPE_UINT8:
            ಪಕ್ಷ GGUF_TYPE_INT8:
            ಪಕ್ಷ GGUF_TYPE_UINT16:
            ಪಕ್ಷ GGUF_TYPE_INT16:
            ಪಕ್ಷ GGUF_TYPE_UINT32:
            ಪಕ್ಷ GGUF_TYPE_INT32:
            ಪಕ್ಷ GGUF_TYPE_FLOAT32:
            ಪಕ್ಷ GGUF_TYPE_UINT64:
            ಪಕ್ಷ GGUF_TYPE_INT64:
            ಪಕ್ಷ GGUF_TYPE_FLOAT64:
            ಪಕ್ಷ GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_ಕೀಲಿ().c_str(), kv.get_type(), kv.data.data(), ne);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGUF_TYPE_STRING: {
                std::vector<ಧ್ರುವ ವರ್ಣ *> tmp(ne);
                ಪ್ರತಿ (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_ಕೀಲಿ().c_str(), tmp.data(), ne);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGUF_TYPE_ARRAY:
            ಸ್ವಭಾವ: GGML_ABORT("invalid type");
        }
    }
}

ಶೂನ್ಯ gguf_add_tensor(
             ಸಂರಚನಾ gguf_context * ctx,
        ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    ಯದಿ (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    ಸಂರಚನಾ gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

ಶೂನ್ಯ gguf_set_tensor_type(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * name, ಗಣನಾ ggml_type type) {
    ಧ್ರುವ int64_t tensor_id = gguf_find_tensor(ctx, name);
    ಯದಿ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    ಸಂರಚನಾ ggml_tensor * tensor = &ctx->info[tensor_id].t;
    ಧ್ರುವ size_t  type_size = ggml_type_size(type);
    ಧ್ರುವ int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    ಧ್ರುವ int64_t n_tensors = gguf_get_n_tensors(ctx);
    ಪ್ರತಿ (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

ಶೂನ್ಯ gguf_set_tensor_data(ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * name, ಧ್ರುವ ಶೂನ್ಯ * data) {
    ಧ್ರುವ int64_t tensor_id = gguf_find_tensor(ctx, name);
    ಯದಿ (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (ಶೂನ್ಯ *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

ಸಂರಚನಾ gguf_writer_base {
    size_t written_bytes {0u};

    ಆಭಾಸ ~gguf_writer_base(ಶೂನ್ಯ) = ಸ್ವಭಾವ;

    // we bet on devirtualization
    ಆಭಾಸ ಶೂನ್ಯ ಲಿಖ(int8_t val) = 0;
    ಆಭಾಸ ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ std::vector<int8_t> & val) = 0;
    ಆಭಾಸ ಶೂನ್ಯ write_tensor_data(ಧ್ರುವ ಸಂರಚನಾ gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ T & val) {
        ಪ್ರತಿ (size_t i = 0; i < ಪರಿಮಾಣ(val); ++i) {
            ಲಿಖ(reinterpret_cast<ಧ್ರುವ int8_t *>(&val)[i]);
        }
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ ತಾರ್ಕಿಕ & val) {
        ಧ್ರುವ int8_t val8 = val ? 1 : 0;
        ಲಿಖ(val8);
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ std::string & val) {
        {
            ಧ್ರುವ uint64_t n = val.length();
            ಲಿಖ(n);
        }
        ಪ್ರತಿ (size_t i = 0; i < val.length(); ++i) {
            ಲಿಖ((val.data())[i]);
        }
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ ವರ್ಣ * val) {
        ಲಿಖ(std::string(val));
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ ಗಣನಾ ggml_type & val) {
        ಲಿಖ(int32_t(val));
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ ಗಣನಾ gguf_type & val) {
        ಲಿಖ(int32_t(val));
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ ಸಂರಚನಾ gguf_kv & kv) {
        ಧ್ರುವ uint64_t ne = kv.get_ne();

        ಲಿಖ(kv.get_ಕೀಲಿ());

        ಯದಿ (kv.is_array) {
            ಲಿಖ(GGUF_TYPE_ARRAY);
            ಲಿಖ(kv.get_type());
            ಲಿಖ(ne);
        } ಅನ್ಯಥಾ {
            ಲಿಖ(kv.get_type());
        }

        ವಿಕಲ್ಪ (kv.get_type()) {
            ಪಕ್ಷ GGUF_TYPE_UINT8:
            ಪಕ್ಷ GGUF_TYPE_INT8:
            ಪಕ್ಷ GGUF_TYPE_UINT16:
            ಪಕ್ಷ GGUF_TYPE_INT16:
            ಪಕ್ಷ GGUF_TYPE_UINT32:
            ಪಕ್ಷ GGUF_TYPE_INT32:
            ಪಕ್ಷ GGUF_TYPE_FLOAT32:
            ಪಕ್ಷ GGUF_TYPE_UINT64:
            ಪಕ್ಷ GGUF_TYPE_INT64:
            ಪಕ್ಷ GGUF_TYPE_FLOAT64: {
                ಲಿಖ(kv.data);
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGUF_TYPE_BOOL: {
                ಪ್ರತಿ (size_t i = 0; i < ne; ++i) {
                    ಲಿಖ(kv.get_val<ತಾರ್ಕಿಕ>(i));
                }
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGUF_TYPE_STRING: {
                ಪ್ರತಿ (size_t i = 0; i < ne; ++i) {
                    ಲಿಖ(kv.get_val<std::string>(i));
                }
            } ಭಙ್ಗ;
            ಪಕ್ಷ GGUF_TYPE_ARRAY:
            ಸ್ವಭಾವ: GGML_ABORT("invalid type");
        }
    }

    ಶೂನ್ಯ write_tensor_meta(ಧ್ರುವ ಸಂರಚನಾ gguf_tensor_info & info) {
        ಲಿಖ(info.t.name);

        ಧ್ರುವ uint32_t n_dims = ggml_n_dims(&info.t);
        ಲಿಖ(n_dims);

        ಪ್ರತಿ (uint32_t j = 0; j < n_dims; ++j) {
            ಲಿಖ(info.t.ne[j]);
        }
        ಲಿಖ(info.t.type);
        ಲಿಖ(info.offset);
    }

    ಶೂನ್ಯ pad(ಧ್ರುವ size_t alignment) {
        ಯಾವತ್ (written_bytes % alignment != 0) {
            ಧ್ರುವ int8_t zero = 0;
            ಲಿಖ(zero);
        }
    }
};

// vector buffer based writer
ಸಂರಚನಾ gguf_writer_buf final : ಸಾರ್ವಜನಿಕ gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::ಲಿಖ;

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    ಶೂನ್ಯ write_tensor_data(ಧ್ರುವ ಸಂರಚನಾ gguf_tensor_info & info, ಧ್ರುವ size_t offset_data, ಧ್ರುವ size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ಧ್ರುವ size_t offset = buf.size();
        ಧ್ರುವ size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        ಯದಿ (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } ಅನ್ಯಥಾ {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
ಸಂರಚನಾ gguf_writer_file final : ಸಾರ್ವಜನಿಕ gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::ಲಿಖ;

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ int8_t val) override {
        ಧ್ರುವ auto real_val = static_cast<uint8_t>(val);
        ಧ್ರುವ auto ret = fputc(real_val, file);
        written_bytes++;
        ಯದಿ (ret != real_val) {
            ಕ್ಷಿಪ std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((ಪೂರ್ಣಾಙ್ಕ)real_val) + "'");
        }
    }

    ಶೂನ್ಯ ಲಿಖ(ಧ್ರುವ std::vector<int8_t> & val) override {
        ಧ್ರುವ auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        ಯದಿ (ret != val.size()) {
            ಕ್ಷಿಪ std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    ಶೂನ್ಯ write_tensor_data(ಧ್ರುವ ಸಂರಚನಾ gguf_tensor_info & info, ಧ್ರುವ size_t offset_data, ಧ್ರುವ size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        ಧ್ರುವ size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        ಯದಿ (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } ಅನ್ಯಥಾ {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        ಲಿಖ(buf);

        pad(alignment);
    }
};

template <typename writer_t>
ಸ್ಥಿರ ಶೂನ್ಯ gguf_write_out(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, writer_t & gw, ತಾರ್ಕಿಕ only_meta) {
    ಧ್ರುವ int64_t n_kv      = gguf_get_n_kv(ctx);
    ಧ್ರುವ int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.ಲಿಖ(GGUF_MAGIC[0]);
    gw.ಲಿಖ(GGUF_MAGIC[1]);
    gw.ಲಿಖ(GGUF_MAGIC[2]);
    gw.ಲಿಖ(GGUF_MAGIC[3]);
    gw.ಲಿಖ(ctx->version);
    gw.ಲಿಖ(n_tensors);
    gw.ಲಿಖ(n_kv);

    // write key-value pairs
    ಪ್ರತಿ (int64_t i = 0; i < n_kv; ++i) {
        gw.ಲಿಖ(ctx->kv[i]);
    }

    // write tensor info
    ಪ್ರತಿ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    ಯದಿ (only_meta) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ size_t offset_data = gw.written_bytes;

    // write tensor data
    ಪ್ರತಿ (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

ಶೂನ್ಯ gguf_write_to_buf(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, std::vector<int8_t> & buf, ತಾರ್ಕಿಕ only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

ತಾರ್ಕಿಕ gguf_write_to_file_ptr(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, FILE * file, ತಾರ್ಕಿಕ only_meta) {
    GGML_ASSERT(file);

    ಯತ್ನ {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } ಗೃಹಾಣ (ಧ್ರುವ std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ gguf_write_to_file(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, ಧ್ರುವ ವರ್ಣ * fname, ತಾರ್ಕಿಕ only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    ಯದಿ (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ success = gguf_write_to_file_ptr(ctx, file, only_meta);
    ಯದಿ (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    ಪ್ರತ್ಯಾಗಮ success;
}

size_t gguf_get_meta_size(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ ಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ buf.size();
}

ಶೂನ್ಯ gguf_get_meta_data(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, ಶೂನ್ಯ * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ ಸತ್ಯ);
    memcpy(data, buf.data(), buf.size());
}
