#pragma once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <vector>

bool ggml_openvino_model_cache_only();
શૂન્ય ggml_openvino_model_cache_init();

સંરચના ggml_openvino_source_mapping {
    uintptr_t begin;
    uintptr_t end;
    uint64_t offset;
    uint64_t identity;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_source_fingerprint(અચલ શૂન્ય * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings);
uint64_t ggml_backend_openvino_weight_fingerprint(અચલ ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_temp_path(અચલ std::string & path);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(અચલ ggml_cgraph * cgraph,
                                         અચલ std::string & device,
                                         bool fa,
                                         અચલ int32_t * rope_params,
                                         પૂર્ણાંક rope_len,
                                         uint64_t extra_cfg,
                                         અચલ std::string & graph_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(અચલ std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(અચલ std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(અચલ std::string & path,
                                              અચલ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              અચલ std::vector<std::string> & inputs,
                                              અચલ std::vector<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(અચલ std::string & path,
                                               અચલ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs);
