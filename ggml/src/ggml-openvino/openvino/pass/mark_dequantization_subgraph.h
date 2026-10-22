// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's ov::pass::MarkDequantization pass declaration.
//
// The pass body is provided by the linked libopenvino.so; only the declaration is needed here so
// we can register it directly in our own TranslateSession::apply_transformations (same approach as
// MarkCompressedFloatConstants's local mirror in mark_decompression_convert_constant_folding.h). This
// lets us mark our GatherMatmul dequantization chain with disable_constant_folding regardless of the
// CPU/GPU plugin's own is_decompression_multiply() consumer allowlist.
// The class layout must stay in sync with
//   openvino/src/common/transformations/include/transformations/low_precision/mark_dequantization_subgraph.hpp

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/core/type/element_type.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/core/visibility.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/pass/matcher_pass.hpp"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ਨਹੀਂਤਾਂ
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ਨਹੀਂਤਾਂ
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#ਜੇ_ਅੰਤ      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkDequantization;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkDequantization : ਜਨਤਕ MatcherPass {
ਜਨਤਕ:
    OPENVINO_MATCHER_PASS_RTTI("MarkDequantization")
    explicit MarkDequantization(ਅਚਲ element::Type_ਸਦਿਸ਼ & precisions,
                                bool fold_ਘਟਾਓ_const = false,
                                bool fold_multiply_const = true);
};
