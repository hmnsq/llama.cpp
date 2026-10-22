#هدايت once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#ضمن "ggml.h"

#ضمن <cstdint>
#ضمن <string>
#ضمن <vector>

bool ggml_openvino_model_cache_only();
فراغ ggml_openvino_model_cache_init();

بنية ggml_openvino_ذريعو_mapping {
    uintptr_t begin;
    uintptr_t پڇاڙي;
    uint64_t وٿي;
    uint64_t سڃاڻپ_عنصر;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_ذريعو_fingerprint(ثابت فراغ * معلومات, size_t سائيز, std::ويڪٽر<ggml_openvino_ذريعو_mapping> & mappings);
uint64_t ggml_backend_openvino_وزن_fingerprint(ثابت ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_عارضي_رستو(ثابت std::string & رستو);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         bool fa,
                                         ثابت int32_t * rope_params,
                                         صحيح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & گراف_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_رستو(ثابت std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_رستو(ثابت std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ثابت std::string & رستو,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::ويڪٽر<std::string> & inputs,
                                              ثابت std::ويڪٽر<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ثابت std::string & رستو,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::ويڪٽر<std::string> & inputs,
                                               std::ويڪٽر<std::string> & outputs);
