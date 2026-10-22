#pragma once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>

bool ggml_openvino_model_cache_only();
शून्य ggml_openvino_model_cache_init();

संरचना ggml_openvino_source_mapping {
    uintptr_t begin;
    uintptr_t end;
    uint64_t offset;
    uint64_t identity;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_source_fingerprint(ध्रुव शून्य * data, माप_प्रकार size, std::vector<ggml_openvino_source_mapping> & mappings);
uint64_t ggml_backend_openvino_weight_fingerprint(ध्रुव ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_temp_path(ध्रुव std::string & path);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ध्रुव ggml_cgraph * cgraph,
                                         ध्रुव std::string & device,
                                         bool fa,
                                         ध्रुव int32_t * rope_params,
                                         पूर्णाङ्क rope_len,
                                         uint64_t extra_cfg,
                                         ध्रुव std::string & graph_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(ध्रुव std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(ध्रुव std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ध्रुव std::string & path,
                                              ध्रुव ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ध्रुव std::vector<std::string> & inputs,
                                              ध्रुव std::vector<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ध्रुव std::string & path,
                                               ध्रुव ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs);
