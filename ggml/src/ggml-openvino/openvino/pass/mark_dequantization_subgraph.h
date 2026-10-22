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

#ہدایت once

#شامل "openvino/core/type/element_type.hpp"
#شامل "openvino/core/visibility.hpp"
#شامل "openvino/pass/matcher_pass.hpp"

#اگر_تعریف_شدہ OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ورنہ
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ورنہ
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#اگر_ختم      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkDequantization;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkDequantization : عوامی MatcherPass {
عوامی:
    OPENVINO_MATCHER_PASS_RTTI("MarkDequantization")
    explicit MarkDequantization(ثابت element::Type_سمتیہ & precisions,
                                bool fold_تفریق_const = false,
                                bool fold_multiply_const = true);
};
