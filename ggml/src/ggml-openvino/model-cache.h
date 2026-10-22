#ہدایت once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#شامل "ggml.h"

#شامل <cstdint>
#شامل <string>
#شامل <vector>

bool ggml_openvino_model_cache_only();
خالی ggml_openvino_model_cache_init();

ساخت ggml_openvino_ماخذ_mapping {
    uintptr_t begin;
    uintptr_t انتہا;
    uint64_t انحراف;
    uint64_t شناختی_عنصر;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_ماخذ_fingerprint(ثابت خالی * ڈیٹا, size_t سائز, std::سمتیہ<ggml_openvino_ماخذ_mapping> & mappings);
uint64_t ggml_backend_openvino_وزن_fingerprint(ثابت ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_عارضی_راستہ(ثابت std::string & راستہ);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         bool fa,
                                         ثابت int32_t * rope_params,
                                         صحیح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & گراف_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_راستہ(ثابت std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_راستہ(ثابت std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ثابت std::string & راستہ,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::سمتیہ<std::string> & inputs,
                                              ثابت std::سمتیہ<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ثابت std::string & راستہ,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::سمتیہ<std::string> & inputs,
                                               std::سمتیہ<std::string> & outputs);
