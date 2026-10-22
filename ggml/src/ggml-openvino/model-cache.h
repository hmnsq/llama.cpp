#pragma once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <vector>

bool ggml_openvino_model_cache_only();
ଶୂନ୍ୟ ggml_openvino_model_cache_init();

ସଂରଚନା ggml_openvino_source_mapping {
    uintptr_t begin;
    uintptr_t end;
    uint64_t offset;
    uint64_t identity;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_source_fingerprint(ଅଚଳ ଶୂନ୍ୟ * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings);
uint64_t ggml_backend_openvino_weight_fingerprint(ଅଚଳ ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_temp_path(ଅଚଳ std::string & path);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ଅଚଳ ggml_cgraph * cgraph,
                                         ଅଚଳ std::string & device,
                                         bool fa,
                                         ଅଚଳ int32_t * rope_params,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା rope_len,
                                         uint64_t extra_cfg,
                                         ଅଚଳ std::string & graph_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(ଅଚଳ std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(ଅଚଳ std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ଅଚଳ std::string & path,
                                              ଅଚଳ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ଅଚଳ std::vector<std::string> & inputs,
                                              ଅଚଳ std::vector<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ଅଚଳ std::string & path,
                                               ଅଚଳ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs);
