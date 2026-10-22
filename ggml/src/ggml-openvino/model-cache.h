#pragma once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>

bool ggml_openvino_model_cache_only();
வெறுமை ggml_openvino_model_cache_init();

கட்டமைப்பு ggml_openvino_source_mapping {
    uintptr_t begin;
    uintptr_t end;
    uint64_t offset;
    uint64_t identity;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_source_fingerprint(மாறிலி வெறுமை * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings);
uint64_t ggml_backend_openvino_weight_fingerprint(மாறிலி ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_temp_path(மாறிலி std::string & path);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(மாறிலி ggml_cgraph * cgraph,
                                         மாறிலி std::string & device,
                                         bool fa,
                                         மாறிலி int32_t * rope_params,
                                         முழு rope_len,
                                         uint64_t extra_cfg,
                                         மாறிலி std::string & graph_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(மாறிலி std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(மாறிலி std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(மாறிலி std::string & path,
                                              மாறிலி ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              மாறிலி std::vector<std::string> & inputs,
                                              மாறிலி std::vector<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(மாறிலி std::string & path,
                                               மாறிலி ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs);
