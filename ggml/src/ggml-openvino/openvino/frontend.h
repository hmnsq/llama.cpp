// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#அந்தர்பாவ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
பொது:
    FrontEnd();

    நிலையான std::shared_ptr<Model> convert(மாறிலி உள்ளீடு_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
