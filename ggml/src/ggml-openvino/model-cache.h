#ਹਦਾਇਤ once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

bool ggml_openvino_model_cache_only();
ਖਾਲੀ ggml_openvino_model_cache_init();

ਬਣਤਰ ggml_openvino_ਸਰੋਤ_mapping {
    uintptr_t begin;
    uintptr_t ਸਿਰਾ;
    uint64_t ਵਿੱਥ;
    uint64_t ਪਛਾਣ_ਤੱਤ;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_ਸਰੋਤ_fingerprint(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ, std::ਸਦਿਸ਼<ggml_openvino_ਸਰੋਤ_mapping> & mappings);
uint64_t ggml_backend_openvino_ਭਾਰ_fingerprint(ਅਚਲ ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_ਆਰਜ਼ੀ_ਰਸਤਾ(ਅਚਲ std::string & ਰਸਤਾ);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ਅਚਲ ggml_cgraph * cgraph,
                                         ਅਚਲ std::string & device,
                                         bool fa,
                                         ਅਚਲ int32_t * rope_params,
                                         ਪੂਰਨ rope_len,
                                         uint64_t extra_cfg,
                                         ਅਚਲ std::string & ਗ੍ਰਾਫ_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_ਰਸਤਾ(ਅਚਲ std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_ਰਸਤਾ(ਅਚਲ std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ਅਚਲ std::string & ਰਸਤਾ,
                                              ਅਚਲ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ਅਚਲ std::ਸਦਿਸ਼<std::string> & inputs,
                                              ਅਚਲ std::ਸਦਿਸ਼<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ਅਚਲ std::string & ਰਸਤਾ,
                                               ਅਚਲ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::ਸਦਿਸ਼<std::string> & inputs,
                                               std::ਸਦਿਸ਼<std::string> & outputs);
