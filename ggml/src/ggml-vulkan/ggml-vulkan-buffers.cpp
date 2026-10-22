#ಅನ್ತರ್ಭಾವ "ggml-vulkan-common.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <shared_mutex>

ಸ್ಥಿರ ಶೂನ್ಯ * portable_aligned_malloc(size_t align, size_t size) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಹಿಂತಿರುಗಿ _aligned_malloc(size, align);
#ಇಲ್ಲದಿದ್ದರೆ
    ಶೂನ್ಯ * ptr = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (posix_memalign(&ptr, align, size) != 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ ptr;
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ಶೂನ್ಯ portable_aligned_free(ಶೂನ್ಯ * ptr) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    _aligned_free(ptr);
#ಇಲ್ಲದಿದ್ದರೆ
    std::ಬಿಡುಗಡೆ(ptr);
#ಆದರೆ_ಅಂತ್ಯ
}

ggml_backend_ಬಫರ್_type_i ggml_backend_vk_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_vk_ಬಫರ್_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_vk_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_vk_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ಖಾಲಿ,
};

ಸ್ಥಿರ std::vector<uint32_t> ggml_vk_ಕಂಡುಹಿಡಿ_ಸ್ಮರಣೆ_properties(ಸ್ಥಿರಾಂಕ vk::Physical_Device_ಸ್ಮರಣೆ_Properties* mem_props, vk::ಸ್ಮರಣೆ_Requirements* mem_req, vk::ಸ್ಮರಣೆ_Property_Flags flags) {
    std::vector<uint32_t> indices;

    ಗಾಗಿ (uint32_t i = 0; i < mem_props->ಸ್ಮರಣೆ_Type_ಎಣಿಕೆ; ++i) {
        vk::ಸ್ಮರಣೆ_Type ಸ್ಮರಣೆ_type = mem_props->ಸ್ಮರಣೆ_Types[i];
        ಆದರೆ ((mem_req->ಸ್ಮರಣೆ_Type_ಬಿಟ್ಟುಗಳು & ((uint64_t)1 << i)) &&
            (flags & ಸ್ಮರಣೆ_type.propertyFlags) == flags &&
            mem_props->ಸ್ಮರಣೆ_Heaps[ಸ್ಮರಣೆ_type.ರಾಶಿ_ಸೂಚ್ಯಂಕ].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    ಹಿಂತಿರುಗಿ indices;
}

ಸ್ಥಿರ vk_ಬಫರ್ ggml_vk_create_ಬಫರ್(vk_device& device, size_t size, ಸ್ಥಿರಾಂಕ std::initializer_list<vk::ಸ್ಮರಣೆ_Property_Flags> & req_flags_list,
                                       ಶೂನ್ಯ *import_ptr = ಶೂನ್ಯಸೂಚಕ) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    ಆದರೆ (size > device->max_ಬಫರ್_ಗಾತ್ರ) {
        ಎಸೆ vk::ಹೊರಹೋಗು_ದ_Device_ಸ್ಮರಣೆ_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_ಬಫರ್ buf = std::make_shared<vk_ಬಫರ್_struct>();

    ಆದರೆ (size == 0) {
        buf->size = 0;
        ಹಿಂತಿರುಗಿ buf;
    }

    vk::ಬಫರ್_Usage_Flags usage_flags = vk::ಬಫರ್_Usage_Flag_ಬಿಟ್ಟುಗಳು::e_Storage_ಬಫರ್ | vk::ಬಫರ್_Usage_Flag_ಬಿಟ್ಟುಗಳು::eTransferSrc | vk::ಬಫರ್_Usage_Flag_ಬಿಟ್ಟುಗಳು::eTransferDst;
    vk::ಸ್ಮರಣೆ_Allocate_Flags mem_flags {};
    ಆದರೆ (device->ಬಫರ್_device_ವಿಳಾಸ) {
        usage_flags |= vk::ಬಫರ್_Usage_Flag_ಬಿಟ್ಟುಗಳು::e_Shader_Device_ವಿಳಾಸ;
        mem_flags |= vk::ಸ್ಮರಣೆ_Allocate_Flag_ಬಿಟ್ಟುಗಳು::e_Device_ವಿಳಾಸ;
    }

    vk::ಬಫರ್_Create_Info ಬಫರ್_create_info{
        vk::ಬಫರ್_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_ಬಹುಲಕ::eExclusive,
        0,
        ಶೂನ್ಯಸೂಚಕ,
    };

    vk::External_ಸ್ಮರಣೆ_ಬಫರ್_Create_Info external_ಸ್ಮರಣೆ_bci;
    ಆದರೆ (import_ptr) {
        external_ಸ್ಮರಣೆ_bci.handleTypes = vk::External_ಸ್ಮರಣೆ_Handle_Type_Flag_ಬಿಟ್ಟುಗಳು::e_Host_ಹಂಚಿಕೆ_EXT;
        ಬಫರ್_create_info.ಗಣ_PNext(&external_ಸ್ಮರಣೆ_bci);
    }

    buf->ಬಫರ್ = device->device.create_ಬಫರ್(ಬಫರ್_create_info);

    vk::ಸ್ಮರಣೆ_Requirements mem_req = device->device.get_ಬಫರ್_ಸ್ಮರಣೆ_Requirements(buf->ಬಫರ್);

    vk::Physical_Device_ಸ್ಮರಣೆ_Properties mem_props = device->physical_device.get_ಸ್ಮರಣೆ_Properties();

    ಸ್ಥಿರಾಂಕ vk::ಸ್ಮರಣೆ_ಆದ್ಯತೆ_Allocate_Info_EXT mem_ಆದ್ಯತೆ_info { 1.0f };

    vk::ಸ್ಮರಣೆ_Allocate_Flags_Info mem_flags_info { mem_flags };

    ಆದರೆ (device->ಸ್ಮರಣೆ_ಆದ್ಯತೆ) {
        mem_flags_info.ಗಣ_PNext(&mem_ಆದ್ಯತೆ_info);
    }

    ಆದರೆ (import_ptr) {
        vk::ಸ್ಮರಣೆ_Host_ಸೂಚಕ_Properties_EXT host_ಸೂಚಕ_props;
        ಪ್ರಯತ್ನಿಸು {
            host_ಸೂಚಕ_props = device->device.get_ಸ್ಮರಣೆ_Host_ಸೂಚಕ_Properties_EXT(vk::External_ಸ್ಮರಣೆ_Handle_Type_Flag_ಬಿಟ್ಟುಗಳು::e_Host_ಹಂಚಿಕೆ_EXT, import_ptr);
        } ಹಿಡಿ (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_ಬಫರ್(buf->ಬಫರ್);
            ಹಿಂತಿರುಗಿ {};
        }
        vk::Physical_Device_ಸ್ಮರಣೆ_Properties mem_props = device->physical_device.get_ಸ್ಮರಣೆ_Properties();

        uint32_t ಸ್ಮರಣೆ_type_idx;
        vk::ಸ್ಮರಣೆ_Property_Flags property_flags = *req_flags_list.begin();
        ಗಾಗಿ (ಸ್ಮರಣೆ_type_idx = 0; ಸ್ಮರಣೆ_type_idx < 32; ++ಸ್ಮರಣೆ_type_idx) {
            ಆದರೆ (!(host_ಸೂಚಕ_props.ಸ್ಮರಣೆ_Type_ಬಿಟ್ಟುಗಳು & (1u << ಸ್ಮರಣೆ_type_idx))) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (!(mem_req.ಸ್ಮರಣೆ_Type_ಬಿಟ್ಟುಗಳು & (1u << ಸ್ಮರಣೆ_type_idx))) {
                ಮುಂದುವರಿ;
            }

            vk::ಸ್ಮರಣೆ_Type ಸ್ಮರಣೆ_type = mem_props.ಸ್ಮರಣೆ_Types[ಸ್ಮರಣೆ_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            ಆದರೆ ((ಸ್ಮರಣೆ_type.propertyFlags & property_flags) == property_flags) {
                property_flags = ಸ್ಮರಣೆ_type.propertyFlags;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (ಸ್ಮರಣೆ_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_ಬಫರ್(buf->ಬಫರ್);
            ಹಿಂತಿರುಗಿ {};
        }

        buf->ಸ್ಮರಣೆ_property_flags = mem_props.ಸ್ಮರಣೆ_Types[ಸ್ಮರಣೆ_type_idx].propertyFlags;
        ಪ್ರಯತ್ನಿಸು {
            vk::Import_ಸ್ಮರಣೆ_Host_ಸೂಚಕ_Info_EXT import_info;
            import_info.handleType = vk::External_ಸ್ಮರಣೆ_Handle_Type_Flag_ಬಿಟ್ಟುಗಳು::e_Host_ಹಂಚಿಕೆ_EXT;
            import_info.p_Host_ಸೂಚಕ = import_ptr;
            import_info.ಗಣ_PNext(&mem_flags_info);
            buf->device_ಸ್ಮರಣೆ = device->device.allocate_ಸ್ಮರಣೆ({ size, ಸ್ಮರಣೆ_type_idx, &import_info });
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & req_flags = *it;

            ಸ್ಥಿರಾಂಕ std::vector<uint32_t> ಸ್ಮರಣೆ_type_indices = ggml_vk_ಕಂಡುಹಿಡಿ_ಸ್ಮರಣೆ_properties(&mem_props, &mem_req, req_flags);

            ಆದರೆ (ಸ್ಮರಣೆ_type_indices.ಖಾಲಿಗಳು()) {
                ಮುಂದುವರಿ;
            }

            ತಾರ್ಕಿಕ ಮುಗಿದ = ಸುಳ್ಳು;

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ mtype_it = ಸ್ಮರಣೆ_type_indices.begin(); mtype_it != ಸ್ಮರಣೆ_type_indices.end(); mtype_it++) {
                ಪ್ರಯತ್ನಿಸು {
                    buf->device_ಸ್ಮರಣೆ = device->device.allocate_ಸ್ಮರಣೆ({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->ಸ್ಮರಣೆ_property_flags = mem_props.ಸ್ಮರಣೆ_Types[*mtype_it].propertyFlags;
                    ಮುಗಿದ = ನಿಜ;
                    ನಿಲ್ಲಿಸು;
                } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    ಆದರೆ (it + 1 == req_flags_list.end() && mtype_it + 1 == ಸ್ಮರಣೆ_type_indices.end()) {
                        device->device.destroy_ಬಫರ್(buf->ಬಫರ್);
                        ಎಸೆ e;
                    }
                }
            }

            ಆದರೆ (ಮುಗಿದ) {
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆದರೆ (!buf->device_ಸ್ಮರಣೆ) {
        device->device.destroy_ಬಫರ್(buf->ಬಫರ್);
        ಎಸೆ vk::ಹೊರಹೋಗು_ದ_Device_ಸ್ಮರಣೆ_Error("No suitable memory type found");
    }

    buf->ptr = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (import_ptr) {
        buf->ptr = import_ptr;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (buf->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible) {
            buf->ptr = device->device.map_ಸ್ಮರಣೆ(buf->device_ಸ್ಮರಣೆ, 0, VK_ಸಂಪೂರ್ಣ_ಗಾತ್ರ);
        }
    }

    device->device.bind_ಬಫರ್_ಸ್ಮರಣೆ(buf->ಬಫರ್, buf->device_ಸ್ಮರಣೆ, 0);

    buf->device = device;
    buf->size = size;

    ಆದರೆ (device->ಬಫರ್_device_ವಿಳಾಸ) {
        ಸ್ಥಿರಾಂಕ vk::ಬಫರ್_Device_ವಿಳಾಸ_Info ವಿಳಾಸ_Info(buf->ಬಫರ್);
        buf->bda_addr = device->device.get_ಬಫರ್_ವಿಳಾಸ(ವಿಳಾಸ_Info);
    }

    device->ಸ್ಮರಣೆ_logger->log_ಹಂಚಿಕೆ(buf, size);

    ಹಿಂತಿರುಗಿ buf;
}

vk_ಬಫರ್ ggml_vk_create_ಬಫರ್_check(vk_device& device, size_t size, vk::ಸ್ಮರಣೆ_Property_Flags req_flags, vk::ಸ್ಮರಣೆ_Property_Flags fallback_flags) {
    ಪ್ರಯತ್ನಿಸು {
        ಹಿಂತಿರುಗಿ ggml_vk_create_ಬಫರ್(device, size, {req_flags, fallback_flags});
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "ggml_vulkan: " << e.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        ಎಸೆ e;
    }
}

vk_ಬಫರ್ ggml_vk_create_ಬಫರ್_device(vk_device& device, size_t size) {
    vk_ಬಫರ್ buf;
    ಪ್ರಯತ್ನಿಸು {
        ಆದರೆ (device->prefer_host_ಸ್ಮರಣೆ) {
            buf = ggml_vk_create_ಬಫರ್(device, size, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent,
                                                       vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_ಬಫರ್(device, size, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent,
                                                       vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal,
                                                       vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent});
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->disable_host_visible_vidmem) {
            ಆದರೆ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_ಬಫರ್(device, size, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal,
                                                           vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent});
            } ಇಲ್ಲದಿದ್ದರೆ {
                buf = ggml_vk_create_ಬಫರ್(device, size, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // use rebar if available, otherwise fallback to device only visible memory
            ಆದರೆ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_ಬಫರ್(device, size, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent,
                                                           vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal,
                                                           vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent});
            } ಇಲ್ಲದಿದ್ದರೆ {
                buf = ggml_vk_create_ಬಫರ್(device, size, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent,
                                                           vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
            }
        }
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "ggml_vulkan: " << e.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        ಎಸೆ e;
    }

    ಹಿಂತಿರುಗಿ buf;
}

ಶೂನ್ಯ ggml_vk_destroy_ಬಫರ್(vk_ಬಫರ್& buf) {
    ಆದರೆ (buf == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (buf->device != ಶೂನ್ಯಸೂಚಕ) {
        buf->device->ಸ್ಮರಣೆ_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ಸ್ಥಿರ size_t ggml_vk_host_ಹಂಚಿಕೆ_ಕಂಡುಹಿಡಿ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr) {
    ಗಾಗಿ (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = (ಸ್ಥಿರಾಂಕ uint8_t *) vk_instance.host_allocations[i].ತಳ;
        ಆದರೆ (ptr >= ತಳ && ptr < ತಳ + vk_instance.host_allocations[i].size) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    ಹಿಂತಿರುಗಿ SIZE_MAX;
}

ಶೂನ್ಯ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_ಸ್ಮರಣೆ("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    ಆದರೆ (vk_instance.multi_device) {
        ಸ್ಥಿರಾಂಕ size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ಸ್ಥಿರಾಂಕ size_t alloc_ಗಾತ್ರ = (size + align - 1) / align * align;

        ಶೂನ್ಯ * ptr = portable_aligned_malloc(align, alloc_ಗಾತ್ರ);
        ಆದರೆ (ptr == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        vk_host_ಹಂಚಿಕೆ ನಮೂದು;
        ನಮೂದು.ತಳ = ptr;
        ನಮೂದು.size = alloc_ಗಾತ್ರ;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(ನಮೂದು));
        ಹಿಂತಿರುಗಿ ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_ಬಫರ್ buf;
    ಪ್ರಯತ್ನಿಸು {
        buf = ggml_vk_create_ಬಫರ್(dev0, size,
            {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCached,
             vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent});
    } ಹಿಡಿ (vk::SystemError&) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (!(buf->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_ಬಫರ್(buf);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    vk_host_ಹಂಚಿಕೆ ನಮೂದು;
    ನಮೂದು.ತಳ             = buf->ptr;
    ನಮೂದು.size             = size;
    ನಮೂದು.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(ನಮೂದು));
    ಹಿಂತಿರುಗಿ buf->ptr;
}

ಶೂನ್ಯ ggml_vk_host_free(ಶೂನ್ಯ * ptr) {
    ಆದರೆ (ptr == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    VK_LOG_ಸ್ಮರಣೆ("ggml_vk_host_free(" << ptr << ")");

    vk_host_ಹಂಚಿಕೆ ನಮೂದು;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ಸ್ಥಿರಾಂಕ size_t ಸೂಚ್ಯಂಕ = ggml_vk_host_ಹಂಚಿಕೆ_ಕಂಡುಹಿಡಿ(ptr);
        ಆದರೆ (ಸೂಚ್ಯಂಕ == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            ಹಿಂತಿರುಗಿ;
        }
        ನಮೂದು = std::move(vk_instance.host_allocations[ಸೂಚ್ಯಂಕ]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + ಸೂಚ್ಯಂಕ);
    }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & view : ನಮೂದು.views) {
        ggml_vk_destroy_ಬಫರ್(view);
    }
    ಆದರೆ (vk_instance.multi_device) {
        portable_aligned_free(ನಮೂದು.ತಳ);
    }
}

ಶೂನ್ಯ ggml_vk_host_get(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr, vk_ಬಫರ್& buf, size_t& buf_ಸ್ಥಾನಾಂತರ) {
    buf = ಶೂನ್ಯಸೂಚಕ;
    buf_ಸ್ಥಾನಾಂತರ = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ಸ್ಥಿರಾಂಕ size_t ಸೂಚ್ಯಂಕ = ggml_vk_host_ಹಂಚಿಕೆ_ಕಂಡುಹಿಡಿ(ptr);
    ಆದರೆ (ಸೂಚ್ಯಂಕ == SIZE_MAX) {
        ಹಿಂತಿರುಗಿ;
    }
    vk_host_ಹಂಚಿಕೆ & ನಮೂದು = vk_instance.host_allocations[ಸೂಚ್ಯಂಕ];

    ಆದರೆ (ನಮೂದು.views[device->idx] == ಶೂನ್ಯಸೂಚಕ) {
        ಸ್ಥಿರಾಂಕ uint32_t ಬಿಟ್ = 1u << device->idx;
        ಆದರೆ (!vk_instance.multi_device || !vk_instance.device_host_ಬಫರ್_import[device->idx] || (vk_instance.host_import_failed & ಬಿಟ್)) {
            ಹಿಂತಿರುಗಿ;
        }
        GGML_ASSERT(device->external_ಸ್ಮರಣೆ_host);
        // const only propagates from the const context; the pointee is mutable
        ನಮೂದು.views[device->idx] = ggml_vk_ಬಫರ್_from_host_ptr(const_cast<vk_device&>(device), ನಮೂದು.ತಳ, ನಮೂದು.size);
        ಆದರೆ (ನಮೂದು.views[device->idx] == ಶೂನ್ಯಸೂಚಕ) {
            vk_instance.host_import_failed |= ಬಿಟ್;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            ಹಿಂತಿರುಗಿ;
        }
    }

    buf = ನಮೂದು.views[device->idx];
    buf_ಸ್ಥಾನಾಂತರ = (ಸ್ಥಿರಾಂಕ uint8_t *) ptr - (ಸ್ಥಿರಾಂಕ uint8_t *) ನಮೂದು.ತಳ;
}

ಶೂನ್ಯ ggml_vk_ensure_sync_staging_ಬಫರ್(vk_device& device, size_t size) {
    ಆದರೆ (device->sync_staging == ಶೂನ್ಯಸೂಚಕ || device->sync_staging->size < size) {
        VK_LOG_ಸ್ಮರಣೆ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_ಬಫರ್(device->sync_staging);
        device->sync_staging = ggml_vk_create_ಬಫರ್_check(device, size,
            vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCached,
            vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent);
    }
}

ಶೂನ್ಯ ggml_vk_ensure_sync_staging_ಬಫರ್(ggml_backend_vk_context * ctx, size_t size) {
    ಆದರೆ (ctx->sync_staging == ಶೂನ್ಯಸೂಚಕ || ctx->sync_staging->size < size) {
        VK_LOG_ಸ್ಮರಣೆ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_ಬಫರ್(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_ಬಫರ್_check(ctx->device, size,
            vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCached,
            vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_ಬಫರ್_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ತಾರ್ಕಿಕ sync_staging = ಸುಳ್ಳು) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    ಆದರೆ(dst->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::ಸಾಲು_ಅಂತ್ಯ;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_ಬಫರ್ buf = ಶೂನ್ಯಸೂಚಕ;
    size_t buf_ಸ್ಥಾನಾಂತರ = 0;
    ggml_vk_host_get(ctx->device, tensor->ದತ್ತಾಂಶ, buf, buf_ಸ್ಥಾನಾಂತರ);

    ಸ್ಥಿರಾಂಕ uint64_t ne0 = tensor->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne1 = tensor->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne2 = tensor->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne3 = tensor->ne[3];
    ಸ್ಥಿರಾಂಕ uint64_t nb0 = tensor->nb[0];
    ಸ್ಥಿರಾಂಕ uint64_t nb1 = tensor->nb[1];
    ಸ್ಥಿರಾಂಕ uint64_t nb2 = tensor->nb[2];
    ಸ್ಥಿರಾಂಕ uint64_t nb3 = tensor->nb[3];
    ಸ್ಥಿರಾಂಕ ggml_type type = tensor->type;
    ಸ್ಥಿರಾಂಕ uint64_t ts = ggml_type_ಗಾತ್ರ(type);
    ಸ್ಥಿರಾಂಕ uint64_t bs = ggml_blck_ಗಾತ್ರ(type);

    ಸ್ಥಿರಾಂಕ uint64_t dstnb0 = ts;
    ಸ್ಥಿರಾಂಕ uint64_t dstnb1 = dstnb0*(ne0/bs);
    ಸ್ಥಿರಾಂಕ uint64_t dstnb2 = dstnb1*ne1;
    ಸ್ಥಿರಾಂಕ uint64_t dstnb3 = dstnb2*ne2;

    ಸ್ಥಿರಾಂಕ uint64_t ne = ggml_nelements(tensor);

    ಆದರೆ (buf != ಶೂನ್ಯಸೂಚಕ) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::ಬಫರ್_Copy> slices;

        ಗಾಗಿ (uint64_t i3 = 0; i3 < ne3; i3++) {
            ಗಾಗಿ (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                ಆದರೆ (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_ಸ್ಥಾನಾಂತರ + i3*nb3 + i2*nb2, ಸ್ಥಾನಾಂತರ + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಗಾಗಿ (uint64_t i1 = 0; i1 < ne1; i1++) {
                        ಆದರೆ (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_ಸ್ಥಾನಾಂತರ + i3*nb3 + i2*nb2 + i1*nb1, ಸ್ಥಾನಾಂತರ + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಸ್ಥಿರಾಂಕ uint64_t s_off = buf_ಸ್ಥಾನಾಂತರ + i3*nb3 + i2*nb2 + i1*nb1;
                            ಸ್ಥಿರಾಂಕ uint64_t d_off = ಸ್ಥಾನಾಂತರ + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            ಗಾಗಿ (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->ಬಫರ್->buf.copy_ಬಫರ್(buf->ಬಫರ್, dst->ಬಫರ್, slices);
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_ಬಫರ್& staging = ctx->device->sync_staging;
    ಸ್ಥಿರಾಂಕ uint64_t copy_ಗಾತ್ರ = ts*ne/bs;
    ggml_vk_ensure_sync_staging_ಬಫರ್(ctx->device, copy_ಗಾತ್ರ);
    Vk_ಬಫರ್_Copy buf_copy{ 0, ಸ್ಥಾನಾಂತರ, copy_ಗಾತ್ರ };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_ಬಫರ್(subctx->s->ಬಫರ್->buf, (Vk_ಬಫರ್)staging->ಬಫರ್, (Vk_ಬಫರ್)dst->ಬಫರ್, 1, &buf_copy);

    ಗಾಗಿ (uint64_t i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            ಆದರೆ (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ಸ್ಥಿರಾಂಕ uint8_t *) tensor->ದತ್ತಾಂಶ + buf_ಸ್ಥಾನಾಂತರ + i3*nb3 + i2*nb2, dstnb2, &subctx->ಒಳ_memcpys);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (uint64_t i1 = 0; i1 < ne1; i1++) {
                    ಆದರೆ (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ಸ್ಥಿರಾಂಕ uint8_t *) tensor->ದತ್ತಾಂಶ + buf_ಸ್ಥಾನಾಂತರ + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->ಒಳ_memcpys);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಸ್ಥಿರಾಂಕ uint64_t s_off = buf_ಸ್ಥಾನಾಂತರ + i3*nb3 + i2*nb2 + i1*nb1;
                        ಸ್ಥಿರಾಂಕ uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        ಗಾಗಿ (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ಸ್ಥಿರಾಂಕ uint8_t *) tensor->ದತ್ತಾಂಶ + s_off + i0*nb0, dstnb0, &subctx->ಒಳ_memcpys);
                        }
                    }
                }
            }
        }
    }
}

ತಾರ್ಕಿಕ ggml_vk_ಬಫರ್_write_2d_async(vk_context subctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ, ತಾರ್ಕಿಕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << ಅಗಲ << ", " << ಎತ್ತರ << ")");
    // Check if src is pinned memory
    vk_ಬಫರ್ buf = ಶೂನ್ಯಸೂಚಕ;
    size_t buf_ಸ್ಥಾನಾಂತರ = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_ಸ್ಥಾನಾಂತರ);

    ಆದರೆ (buf != ಶೂನ್ಯಸೂಚಕ) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::ಬಫರ್_Copy> slices(1);
        ಆದರೆ (ಅಗಲ == spitch && ಅಗಲ == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_ಸ್ಥಾನಾಂತರ_2 = buf_ಸ್ಥಾನಾಂತರ;
            slices[0].dst_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ;
            slices[0].size = ಅಗಲ * ಎತ್ತರ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            slices.resize(ಎತ್ತರ);
            ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
                slices[i].src_ಸ್ಥಾನಾಂತರ_2 = buf_ಸ್ಥಾನಾಂತರ + i * spitch;
                slices[i].dst_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ + i * dpitch;
                slices[i].size = ಅಗಲ;
            }
        }

        ggml_vk_sync_buffers(ಶೂನ್ಯಸೂಚಕ, subctx);
        subctx->s->ಬಫರ್->buf.copy_ಬಫರ್(buf->ಬಫರ್, dst->ಬಫರ್, slices);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    VK_LOG_DEBUG("STAGING");

    ಆದರೆ (!sync_staging) {
        // copy was not handled caller needs to fall back
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Staging buffer required
    ಸ್ಥಿರಾಂಕ size_t staging_ಗಾತ್ರ = ಅಗಲ * ಎತ್ತರ;
    ggml_vk_ensure_sync_staging_ಬಫರ್(dst->device, staging_ಗಾತ್ರ);

    vk_ಬಫರ್& staging_ಬಫರ್ = dst->device->sync_staging;

    std::vector<vk::ಬಫರ್_Copy> slices(1);
    ಆದರೆ (ಅಗಲ == dpitch) {
        slices[0].src_ಸ್ಥಾನಾಂತರ_2 = 0;
        slices[0].dst_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ;
        slices[0].size = staging_ಗಾತ್ರ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        slices.resize(ಎತ್ತರ);
        ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
            slices[i].src_ಸ್ಥಾನಾಂತರ_2 = i * ಅಗಲ;
            slices[i].dst_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ + i * dpitch;
            slices[i].size = ಅಗಲ;
        }
    }

    ggml_vk_sync_buffers(ಶೂನ್ಯಸೂಚಕ, subctx);
    subctx->s->ಬಫರ್->buf.copy_ಬಫರ್(staging_ಬಫರ್->ಬಫರ್, dst->ಬಫರ್, slices);

    ಆದರೆ (ಅಗಲ == spitch) {
        deferred_memcpy((uint8_t *)staging_ಬಫರ್->ptr, src, staging_ಗಾತ್ರ, &subctx->ಒಳ_memcpys);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
            deferred_memcpy((uint8_t *)staging_ಬಫರ್->ptr + i * ಅಗಲ, (ಸ್ಥಿರಾಂಕ uint8_t *) src + i * spitch, ಅಗಲ, &subctx->ಒಳ_memcpys);
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_ಬಫರ್_write_async(vk_context subctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t size, ತಾರ್ಕಿಕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    ಹಿಂತಿರುಗಿ ggml_vk_ಬಫರ್_write_2d_async(subctx, dst, ಸ್ಥಾನಾಂತರ, src, size, size, size, 1, sync_staging);
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_write_2d(vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << ಅಗಲ << ", " << ಎತ್ತರ << ")");
    // Buffer is already mapped
    ಆದರೆ(dst->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible) {
        GGML_ASSERT(dst->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent);

        ಆದರೆ (ಅಗಲ == spitch && ಅಗಲ == dpitch) {
            ಸ್ಮರಣೆ_ನಕಲು((uint8_t *)dst->ptr + ಸ್ಥಾನಾಂತರ, src, ಅಗಲ * ಎತ್ತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
                ಸ್ಮರಣೆ_ನಕಲು((uint8_t *)dst->ptr + ಸ್ಥಾನಾಂತರ + i * dpitch, (ಸ್ಥಿರಾಂಕ uint8_t *) src + i * spitch, ಅಗಲ);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_ಸರತಿ->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        ತಾರ್ಕಿಕ ret = ggml_vk_ಬಫರ್_write_2d_async(subctx, dst, ಸ್ಥಾನಾಂತರ, src, spitch, dpitch, ಅಗಲ, ಎತ್ತರ, ನಿಜ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_ಕೊನೆ(subctx);

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : subctx->ಒಳ_memcpys) {
            ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
        }

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ನಿಜ, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_ಸರತಿ_command_pools_cleanup(dst->device);
    }
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_write(vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_ಬಫರ್_write_2d(dst, ಸ್ಥಾನಾಂತರ, src, size, size, size, 1);
}

ತಾರ್ಕಿಕ ggml_vk_ಬಫರ್_read_2d_async(vk_context subctx, vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ, ತಾರ್ಕಿಕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << ಸ್ಥಾನಾಂತರ << ", width=" << ಅಗಲ << ", height=" << ಎತ್ತರ << ")");
    GGML_ASSERT(ಅಗಲ > 0);
    GGML_ASSERT(ಎತ್ತರ > 0);
    GGML_ASSERT(src != ಶೂನ್ಯಸೂಚಕ);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_ಬಫರ್ buf = ಶೂನ್ಯಸೂಚಕ;
    size_t buf_ಸ್ಥಾನಾಂತರ = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_ಸ್ಥಾನಾಂತರ);

    std::vector<vk::ಬಫರ್_Copy> slices(1);
    ಆದರೆ (ಅಗಲ == spitch && ಅಗಲ == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ;
        slices[0].dst_ಸ್ಥಾನಾಂತರ_2 = buf_ಸ್ಥಾನಾಂತರ;
        slices[0].size = ಅಗಲ * ಎತ್ತರ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        slices.resize(ಎತ್ತರ);
        ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
            slices[i].src_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ + i * spitch;
            slices[i].dst_ಸ್ಥಾನಾಂತರ_2 = buf_ಸ್ಥಾನಾಂತರ + i * dpitch;
            slices[i].size = ಅಗಲ;
        }
    }

    ಆದರೆ (buf != ಶೂನ್ಯಸೂಚಕ) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(ಶೂನ್ಯಸೂಚಕ, subctx);
        subctx->s->ಬಫರ್->buf.copy_ಬಫರ್(src->ಬಫರ್, buf->ಬಫರ್, slices);

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    VK_LOG_DEBUG("STAGING");

    ಆದರೆ (!sync_staging) {
        // copy was not handled caller needs to fall back
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Fall back to staging buffer
    ಸ್ಥಿರಾಂಕ size_t staging_ಗಾತ್ರ = ಅಗಲ * ಎತ್ತರ;
    ggml_vk_ensure_sync_staging_ಬಫರ್(src->device, staging_ಗಾತ್ರ);

    vk_ಬಫರ್& staging_ಬಫರ್ = src->device->sync_staging;

    std::vector<vk::ಬಫರ್_Copy> staging_slices(1);
    ಆದರೆ (ಅಗಲ == spitch) {
        staging_slices[0].src_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ;
        staging_slices[0].dst_ಸ್ಥಾನಾಂತರ_2 = 0;
        staging_slices[0].size = staging_ಗಾತ್ರ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        staging_slices.resize(ಎತ್ತರ);
        ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
            staging_slices[i].src_ಸ್ಥಾನಾಂತರ_2 = ಸ್ಥಾನಾಂತರ + i * spitch;
            staging_slices[i].dst_ಸ್ಥಾನಾಂತರ_2 = i * ಅಗಲ;
            staging_slices[i].size = ಅಗಲ;
        }
    }

    ggml_vk_sync_buffers(ಶೂನ್ಯಸೂಚಕ, subctx);
    subctx->s->ಬಫರ್->buf.copy_ಬಫರ್(src->ಬಫರ್, staging_ಬಫರ್->ಬಫರ್, staging_slices);

    ಆದರೆ (ಅಗಲ == dpitch) {
        deferred_memcpy(dst, staging_ಬಫರ್->ptr, staging_ಗಾತ್ರ, &subctx->ಹೊರಹೋಗು_memcpys);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ಸ್ಥಿರಾಂಕ uint8_t *) staging_ಬಫರ್->ptr + i * ಅಗಲ, ಅಗಲ, &subctx->ಹೊರಹೋಗು_memcpys);
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_ಬಫರ್_read_async(vk_context subctx, vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t size, ತಾರ್ಕಿಕ sync_staging = ಸುಳ್ಳು) {
    ಹಿಂತಿರುಗಿ ggml_vk_ಬಫರ್_read_2d_async(subctx, src, ಸ್ಥಾನಾಂತರ, dst, size, size, size, 1, sync_staging);
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_read_2d(vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->ಬಫರ್ << ", " << ಸ್ಥಾನಾಂತರ << ", " << ಅಗಲ << ", " << ಎತ್ತರ << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCached);
    ಆದರೆ(src->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_ಸರತಿ->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->ಬಫರ್->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eComputeShader | vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eTransfer,
            vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eHost,
            {},
            { { vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eShaderWrite | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferWrite,
                vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eHostRead } },
            {}, {});
        ggml_vk_ctx_ಕೊನೆ(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ನಿಜ, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_ಸರತಿ_command_pools_cleanup(src->device);

        ಆದರೆ (ಅಗಲ == spitch && ಅಗಲ == dpitch) {
            ಸ್ಮರಣೆ_ನಕಲು(dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src->ptr + ಸ್ಥಾನಾಂತರ, ಅಗಲ * ಎತ್ತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (size_t i = 0; i < ಎತ್ತರ; i++) {
                ಸ್ಮರಣೆ_ನಕಲು((uint8_t *) dst + i * dpitch, (ಸ್ಥಿರಾಂಕ uint8_t *) src->ptr + ಸ್ಥಾನಾಂತರ + i * spitch, ಅಗಲ);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_ಸರತಿ->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ತಾರ್ಕಿಕ ret = ggml_vk_ಬಫರ್_read_2d_async(subctx, src, ಸ್ಥಾನಾಂತರ, dst, spitch, dpitch, ಅಗಲ, ಎತ್ತರ, ನಿಜ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_ಕೊನೆ(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ನಿಜ, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_ಸರತಿ_command_pools_cleanup(src->device);

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : subctx->ಹೊರಹೋಗು_memcpys) {
            ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
        }
    }
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_read(vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->ಬಫರ್ << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");
    ggml_vk_ಬಫರ್_read_2d(src, ಸ್ಥಾನಾಂತರ, dst, size, size, size, 1);
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_copy_async(vk_context& ctx, vk_ಬಫರ್& dst, size_t dst_ಸ್ಥಾನಾಂತರ, vk_ಬಫರ್& src, size_t src_ಸ್ಥಾನಾಂತರ, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_ಬಫರ್_Copy bc{ src_ಸ್ಥಾನಾಂತರ, dst_ಸ್ಥಾನಾಂತರ, size };

    vk_Cmd_Copy_ಬಫರ್(ctx->s->ಬಫರ್->buf, (Vk_ಬಫರ್)src->ಬಫರ್, (Vk_ಬಫರ್)dst->ಬಫರ್, 1, &bc);
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_copy(vk_ಬಫರ್& dst, size_t dst_ಸ್ಥಾನಾಂತರ, vk_ಬಫರ್& src, size_t src_ಸ್ಥಾನಾಂತರ, size_t size) {
    ಆದರೆ (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_ಸರತಿ->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_ಬಫರ್_copy_async(subctx, dst, dst_ಸ್ಥಾನಾಂತರ, src, src_ಸ್ಥಾನಾಂತರ, size);
        ggml_vk_ctx_ಕೊನೆ(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ನಿಜ, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_ಸರತಿ_command_pools_cleanup(src->device);
    } ಇಲ್ಲದಿದ್ದರೆ {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_ಬಫರ್(src->device, size);

        // Copy to src staging buffer
        ggml_vk_ಬಫರ್_copy(src->device->sync_staging, 0, src, src_ಸ್ಥಾನಾಂತರ, size);
        // Copy to dst buffer
        ggml_vk_ಬಫರ್_write(dst, dst_ಸ್ಥಾನಾಂತರ, src->device->sync_staging->ptr, size);
    }
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_memset_async(vk_context& ctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << ಸ್ಥಾನಾಂತರ << ", " << c << ", " << size << ")");

    ಆದರೆ (dst->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + ಸ್ಥಾನಾಂತರ, c, size, &ctx->memsets);
        ಹಿಂತಿರುಗಿ;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->ಬಫರ್->buf.fill_ಬಫರ್(dst->ಬಫರ್, ಸ್ಥಾನಾಂತರ, size, c);
}

ಶೂನ್ಯ ggml_vk_ಬಫರ್_memset(vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << ಸ್ಥಾನಾಂತರ << ", " << c << ", " << size << ")");

    ಆದರೆ (dst->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + ಸ್ಥಾನಾಂತರ, c, size);
        ಹಿಂತಿರುಗಿ;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_ಸರತಿ->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->ಬಫರ್->buf.fill_ಬಫರ್(dst->ಬಫರ್, ಸ್ಥಾನಾಂತರ, size, c);
    ggml_vk_ctx_ಕೊನೆ(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ನಿಜ, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_ಸರತಿ_command_pools_cleanup(dst->device);
}

ggml_backend_ಬಫರ್_i ggml_backend_vk_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_vk_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_vk_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_vk_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_vk_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_ಬಫರ್_ಗಣ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_ಬಫರ್_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

vk_ಬಫರ್ ggml_vk_ಬಫರ್_from_host_ptr(vk_device & device, ಶೂನ್ಯ * ptr, size_t size) {
    ಆದರೆ (!device->external_ಸ್ಮರಣೆ_host) {
        ಹಿಂತಿರುಗಿ {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    ಆದರೆ (uptr & (device->min_imported_host_ಸೂಚಕ_alignment - 1)) {
        ಹಿಂತಿರುಗಿ {};
    }
    ಆದರೆ (size & (device->min_imported_host_ಸೂಚಕ_alignment - 1)) {
        ಹಿಂತಿರುಗಿ {};
    }

    ಸ್ಥಿರಾಂಕ vk::ಸ್ಮರಣೆ_Property_Flags property_flags = vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCached;

    vk_ಬಫರ್ buf {};
    ಪ್ರಯತ್ನಿಸು {
        buf = ggml_vk_create_ಬಫರ್(device, size, { property_flags }, ptr);
    } ಹಿಡಿ (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    ಹಿಂತಿರುಗಿ buf;
}

