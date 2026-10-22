#لارښوونه once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#ضمن "ggml.h"

#ضمن <cstdint>
#ضمن <string>
#ضمن <vector>

bool ggml_openvino_model_cache_only();
فراغ ggml_openvino_model_cache_init();

بنية ggml_openvino_سرچینه_mapping {
    uintptr_t begin;
    uintptr_t پای_ټکی;
    uint64_t دننه_واټن;
    uint64_t بې_اغېزه_عنصر;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_سرچینه_fingerprint(ثابت فراغ * معلومات, size_t اندازه, std::ویکټور<ggml_openvino_سرچینه_mapping> & mappings);
uint64_t ggml_backend_openvino_وزن_fingerprint(ثابت ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_لنډمهاله_لار(ثابت std::string & لار);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         bool fa,
                                         ثابت int32_t * rope_params,
                                         صحيح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & ګراف_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_لار(ثابت std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_لار(ثابت std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(ثابت std::string & لار,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::ویکټور<std::string> & inputs,
                                              ثابت std::ویکټور<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(ثابت std::string & لار,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::ویکټور<std::string> & inputs,
                                               std::ویکټور<std::string> & outputs);
