#pragma once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>

bool ggml_openvino_model_cache_only();
শূন্য ggml_openvino_model_cache_init();

কাঠামো ggml_openvino_source_mapping {
    uintptr_t begin;
    uintptr_t end;
    uint64_t offset;
    uint64_t identity;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_source_fingerprint(ধ্রুবক শূন্য * তথ্য, size_t আকার_2, std::vector<ggml_openvino_source_mapping> & mappings);
uint64_t ggml_backend_openvino_weight_fingerprint(ধ্রুবক ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_অস্থায়ী_path(ধ্রুবক std::string & path);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ধ্রুবক ggml_cgraph * cgraph,
                                         ধ্রুবক std::string & device,
                                         bool fa,
                                         ধ্রুবক int32_t * rope_params,
                                         পূর্ণসংখ্যা rope_len,
                                         uint64_t extra_cfg,
                                         ধ্রুবক std::string & graph_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(ধ্রুবক std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(ধ্রুবক std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ধ্রুবক std::string & path,
                                              ধ্রুবক ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ধ্রুবক std::vector<std::string> & inputs,
                                              ধ্রুবক std::vector<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ধ্রুবক std::string & path,
                                               ধ্রুবক ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs);
