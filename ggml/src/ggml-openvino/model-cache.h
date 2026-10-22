#సూచన once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.

#అన్తర్భావ "ggml.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>

bool ggml_openvino_model_cache_only();
శూన్య ggml_openvino_model_cache_init();

నిర్మాణం ggml_openvino_ఆధారం_mapping {
    uintptr_t begin;
    uintptr_t చివర;
    uint64_t స్థానభ్రంశం;
    uint64_t తటస్థం;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_ఆధారం_fingerprint(స్థిరరాశి శూన్య * దత్తాంశం, size_t కొలత, std::సదిశ<ggml_openvino_ఆధారం_mapping> & mappings);
uint64_t ggml_backend_openvino_బరువు_fingerprint(స్థిరరాశి ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_తాత్కాలిక_మార్గం(స్థిరరాశి std::string & మార్గం);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(స్థిరరాశి ggml_cgraph * cgraph,
                                         స్థిరరాశి std::string & device,
                                         bool fa,
                                         స్థిరరాశి int32_t * rope_params,
                                         పూర్ణాంకం rope_len,
                                         uint64_t extra_cfg,
                                         స్థిరరాశి std::string & లేఖాచిత్రం_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_మార్గం(స్థిరరాశి std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_మార్గం(స్థిరరాశి std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(స్థిరరాశి std::string & మార్గం,
                                              స్థిరరాశి ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              స్థిరరాశి std::సదిశ<std::string> & inputs,
                                              స్థిరరాశి std::సదిశ<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(స్థిరరాశి std::string & మార్గం,
                                               స్థిరరాశి ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::సదిశ<std::string> & inputs,
                                               std::సదిశ<std::string> & outputs);
