// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#अन्तर्भाव <openvino/core/core_visibility.hpp>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/runtime_attribute.hpp>

नामाकाश ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
वर्ग OPENVINO_API WeightlessCacheAttribute : सार्वजनिक RuntimeAttribute {
सार्वजनिक:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = delete;

    WeightlessCacheAttribute(माप_प्रकार original_size, माप_प्रकार bin_offset, ov::element::Type original_dtype)
        : original_size(original_size),
          bin_offset(bin_offset),
          original_dtype(original_dtype) {}

    तार्किक is_copyable() स्थिरांक override;

    माप_प्रकार original_size;
    माप_प्रकार bin_offset;
    ov::element::Type original_dtype;
};

}  // namespace ov
