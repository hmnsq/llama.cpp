#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "common.hpp"

// The legacy implementation uses SLM to implement sorting and top_k selection.
// SLM is limited to 128KB on Xe, which limits how much can be sorted to k<32.
// After a k=8, the radix selection becomes beneficial for most cases, because
// scan-merge has (block + 1) * k pairs of (value, index). Given normal sorting of nlog(n),
// radix-select becomes beneficial quite early. This sets it to 8 - however, the other parameters
// (columns and rows) may also be a driving factor.
// We select the legacy implementation for k below this constant because the overhead of radix select
// exceeds the benefit for very small problems
constexpr ಪೂರ್ಣಾಂಕ SYCL_ತುದಿ_K_ಪರಿಶೀಲನೆ_ವಿಲೀನ_MAX_K = 8;

// Top-k of every row of src, k indices per row into dst_indices, in no particular order.
// Picks between the one-group-per-row and the split-row kernel from the shape and the device.
ಶೂನ್ಯ ggml_sycl_ತುದಿ_k_radix(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
    int32_t *       dst_indices,
    ಸ್ಥಿರಾಂಕ int64_t   ncols,
    ಸ್ಥಿರಾಂಕ int64_t   nrows,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k,
    dpct::ಸರತಿ_ptr main_ಹರಿವು);
