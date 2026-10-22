#ಸೂಚನೆ once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

bool ggml_openvino_model_cache_only();
ಶೂನ್ಯ ggml_openvino_model_cache_init();

ರಚನೆ ggml_openvino_ಮೂಲ_mapping {
    uintptr_t begin;
    uintptr_t ಕೊನೆ;
    uint64_t ಸ್ಥಾನಾಂತರ;
    uint64_t ತಟಸ್ಥ;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_ಮೂಲ_fingerprint(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2, std::ಸದಿಶ<ggml_openvino_ಮೂಲ_mapping> & mappings);
uint64_t ggml_backend_openvino_ತೂಕ_fingerprint(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_ತಾತ್ಕಾಲಿಕ_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ std::string & ಮಾರ್ಗ);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                         ಸ್ಥಿರಾಂಕ std::string & device,
                                         bool fa,
                                         ಸ್ಥಿರಾಂಕ int32_t * rope_params,
                                         ಪೂರ್ಣಾಂಕ rope_len,
                                         uint64_t extra_cfg,
                                         ಸ್ಥಿರಾಂಕ std::string & ನಕ್ಷೆ_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ಸ್ಥಿರಾಂಕ std::string & ಮಾರ್ಗ,
                                              ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<std::string> & inputs,
                                              ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ಸ್ಥಿರಾಂಕ std::string & ಮಾರ್ಗ,
                                               ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::ಸದಿಶ<std::string> & inputs,
                                               std::ಸದಿಶ<std::string> & outputs);
