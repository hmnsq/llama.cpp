#അന്തര്ഭാവ <filesystem>
#അന്തര്ഭാവ <set>
#അന്തര്ഭാവ <sstream>
#അന്തര്ഭാവ <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#അല്ലെങ്കിൽ
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "htp-drv.h"
#അന്തര്ഭാവ "libdl.h"

#അന്തര്ഭാവ <domain.h>

//
// Driver API types
//

തരനിർവചനം ശൂന്യം * (*rpcmem_alloc_pfn_t)(പൂർണ്ണസംഖ്യ heapid, uint32_t flags, പൂർണ്ണസംഖ്യ size);
തരനിർവചനം ശൂന്യം * (*rpcmem_alloc2_pfn_t)(പൂർണ്ണസംഖ്യ heapid, uint32_t flags, size_t size);
തരനിർവചനം ശൂന്യം   (*rpcmem_free_pfn_t)(ശൂന്യം * po);
തരനിർവചനം പൂർണ്ണസംഖ്യ    (*rpcmem_to_fd_pfn_t)(ശൂന്യം * po);

തരനിർവചനം AEEResult (*dspqueue_create_pfn_t)(പൂർണ്ണസംഖ്യ                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           ശൂന്യം *              callback_context,
                                           dspqueue_t *        queue);
തരനിർവചനം AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
തരനിർവചനം AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
തരനിർവചനം AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t അക്കം_buffers,
                                          ഘടന dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          സ്ഥിരം uint8_t *message,
                                          uint32_t timeout_us);
തരനിർവചനം AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *അക്കം_buffers,
                                         ഘടന dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
തരനിർവചനം AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *അക്കം_buffers,
                                                 ഘടന dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*fastrpc_mmap_pfn_t)(പൂർണ്ണസംഖ്യ domain, പൂർണ്ണസംഖ്യ fd, ശൂന്യം *addr, പൂർണ്ണസംഖ്യ offset, size_t length, എണ്ണൽ fastrpc_map_flags flags);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*fastrpc_munmap_pfn_t)(പൂർണ്ണസംഖ്യ domain, പൂർണ്ണസംഖ്യ fd, ശൂന്യം *addr, size_t length);

തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_handle64_open_pfn_t)(സ്ഥിരം അക്ഷരം* name, remote_handle64 *ph);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_handle64_close_pfn_t)(remote_handle h);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_handle_control_pfn_t)(uint32_t req, ശൂന്യം* data, uint32_t datalen);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, ശൂന്യം* data, uint32_t datalen);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_session_control_pfn_t)(uint32_t req, ശൂന്യം *data, uint32_t datalen);
തരനിർവചനം പൂർണ്ണസംഖ്യ (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

ശൂന്യം * rpcmem_alloc(പൂർണ്ണസംഖ്യ heapid, uint32_t flags, പൂർണ്ണസംഖ്യ size) {
    തിരികെ rpcmem_alloc_pfn(heapid, flags, size);
}

ശൂന്യം * rpcmem_alloc2(പൂർണ്ണസംഖ്യ heapid, uint32_t flags, size_t size) {
    എങ്കിൽ (rpcmem_alloc2_pfn) {
        തിരികെ rpcmem_alloc2_pfn(heapid, flags, size);
    } അല്ലെങ്കിൽ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        തിരികെ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

ശൂന്യം rpcmem_free(ശൂന്യം * po) {
    തിരികെ rpcmem_free_pfn(po);
}

പൂർണ്ണസംഖ്യ rpcmem_to_fd(ശൂന്യം * po) {
    തിരികെ rpcmem_to_fd_pfn(po);
}

HTPDRV_API പൂർണ്ണസംഖ്യ fastrpc_mmap(പൂർണ്ണസംഖ്യ domain, പൂർണ്ണസംഖ്യ fd, ശൂന്യം * addr, പൂർണ്ണസംഖ്യ offset, size_t length, എണ്ണൽ fastrpc_map_flags flags) {
    തിരികെ fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API പൂർണ്ണസംഖ്യ fastrpc_munmap(പൂർണ്ണസംഖ്യ domain, പൂർണ്ണസംഖ്യ fd, ശൂന്യം * addr, size_t length) {
    തിരികെ fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(പൂർണ്ണസംഖ്യ                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          ശൂന്യം *              callback_context,
                          dspqueue_t *        queue) {
    തിരികെ dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    തിരികെ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    തിരികെ dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 അക്കം_buffers,
                         ഘടന dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         സ്ഥിരം uint8_t *          message,
                         uint32_t                 timeout_us) {
    തിരികെ dspqueue_write_pfn(queue, flags, അക്കം_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               അക്കം_buffers,
                        ഘടന dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    എങ്കിൽ (timeout_us == 0) {
        തിരികെ dspqueue_read_noblock_pfn(queue, flags, max_buffers, അക്കം_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    തിരികെ dspqueue_read_pfn(queue, flags, max_buffers, അക്കം_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_handle64_open(സ്ഥിരം അക്ഷരം * name, remote_handle64 * ph) {
    തിരികെ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    തിരികെ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_handle64_close(remote_handle64 h) {
    തിരികെ remote_handle64_close_pfn(h);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_handle_control(uint32_t req, ശൂന്യം * data, uint32_t datalen) {
    തിരികെ remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_handle64_control(remote_handle64 h, uint32_t req, ശൂന്യം * data, uint32_t datalen) {
    തിരികെ remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_session_control(uint32_t req, ശൂന്യം * data, uint32_t datalen) {
    തിരികെ remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API പൂർണ്ണസംഖ്യ remote_system_request(system_req_payload * req) {
    എങ്കിൽ (!remote_system_request_pfn) {
        തിരികെ AEE_EUNSUPPORTEDAPI;
    }
    തിരികെ remote_system_request_pfn(req);
}

#ifdef _WIN32

സ്ഥിതം std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    എങ്കിൽ (wstr.empty()) {
        തിരികെ result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (പൂർണ്ണസംഖ്യ) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    എങ്കിൽ (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        എറിയുക std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    പൂർണ്ണസംഖ്യ bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (പൂർണ്ണസംഖ്യ) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    എങ്കിൽ (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        എറിയുക std::runtime_error("Wstring conversion failed");
    }
    തിരികെ result;
}

സ്ഥിതം std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ശൂന്യ, ശൂന്യ, STANDARD_RIGHTS_READ);
    എങ്കിൽ (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        തിരികെ result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    എങ്കിൽ (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        തിരികെ result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    എങ്കിൽ (!QueryServiceConfigW(schService, ശൂന്യ, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        തിരികെ result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    എങ്കിൽ (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        തിരികെ result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_ദ്വിമാന_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    സ്ഥിരം std::wstring system_വേര്_Placeholder = L"\\SystemRoot";
    എങ്കിൽ (0 != driverPath.compare(0, system_വേര്_Placeholder.length(), system_വേര്_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        തിരികെ result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    സ്ഥിരം std::wstring system_വേര്_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD അക്കം_വാക്കുകൾ = GetEnvironmentVariableW(system_വേര്_Env.c_str(), ശൂന്യ, 0);
    എങ്കിൽ (അക്കം_വാക്കുകൾ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        തിരികെ result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_വേര്(അക്കം_വാക്കുകൾ + 1);
    അക്കം_വാക്കുകൾ = GetEnvironmentVariableW(system_വേര്_Env.c_str(), system_വേര്.data(), അക്കം_വാക്കുകൾ + 1);
    എങ്കിൽ (അക്കം_വാക്കുകൾ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        തിരികെ result;
    }
    driverPath.replace(0, system_വേര്_Placeholder.length(), std::wstring(system_വേര്.data()));

    തിരികെ wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

പൂർണ്ണസംഖ്യ htpdrv_init() {
    സ്ഥിതം dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    സ്ഥിതം തർക്കം initialized = അസത്യം;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#അല്ലെങ്കിൽ
    std::string drv_path = "libcdsprpc.so";
#endif
    എങ്കിൽ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        തിരികെ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    എങ്കിൽ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        തിരികെ AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, ചിഹ്നം, ignore)                               \
    ചെയ്യുക {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        എങ്കിൽ (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            തിരികെ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ആകുമ്പോൾ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, അസത്യം);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, സത്യം);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, അസത്യം);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, അസത്യം);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, അസത്യം);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, അസത്യം);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, അസത്യം);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, അസത്യം);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, അസത്യം);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, അസത്യം);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, അസത്യം);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, അസത്യം);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, അസത്യം);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, അസത്യം);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, അസത്യം);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, അസത്യം);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, അസത്യം);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, അസത്യം);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, സത്യം);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = സത്യം;

    തിരികെ AEE_SUCCESS;
}

domain * htpdrv_get_domain(പൂർണ്ണസംഖ്യ domain_id) {
    പൂർണ്ണസംഖ്യ i    = 0;
    പൂർണ്ണസംഖ്യ size = വലുപ്പം(supported_domains) / വലുപ്പം(domain);

    വേണ്ടി (i = 0; i < size; i++) {
        എങ്കിൽ (supported_domains[i].id == domain_id) {
            തിരികെ &supported_domains[i];
        }
    }

    തിരികെ ശൂന്യ;
}

പൂർണ്ണസംഖ്യ htpdrv_get_arch(പൂർണ്ണസംഖ്യ domain, പൂർണ്ണസംഖ്യ * arch) {
    എങ്കിൽ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        തിരികെ AEE_EUNSUPPORTEDAPI;
    }

    ഘടന remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    പൂർണ്ണസംഖ്യ err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, വലുപ്പം(arch_ver));
    എങ്കിൽ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        തിരികെ AEE_EUNSUPPORTEDAPI;
    }

    എങ്കിൽ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        തിരികെ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (പൂർണ്ണസംഖ്യ) ((val >> 4) * 10 + (val & 0x0f));
    തിരികെ 0;
}
