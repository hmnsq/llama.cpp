#आशय once

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

#अगर_घोषित _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#पूर्ण_अगर

#समावेश <AEEStdErr.h>
#समावेश <rpcmem.h>
#समावेश <remote.h>
#समावेश <dspqueue.h>

#अगर defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) बाह्य
#    अन्यथा
#        define HTPDRV_API __declspec(dllimport) बाह्य
#    endif
#अन्यथा
#    define HTPDRV_API __attribute__ ((visibility ("default"))) बाह्य
#पूर्ण_अगर

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#अगर_अघोषित DSP_OFFSET
#    define DSP_OFFSET 0x80000400
#पूर्ण_अगर

/* Errno for connection reset by peer. */
#अगर_अघोषित ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#पूर्ण_अगर

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#अगर_अघोषित SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    अन्यथा
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        अन्यथा
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#पूर्ण_अगर

/* Include windows specific header files. */
#अगर_घोषित _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#पूर्ण_अगर

/* Includes and defines for all HLOS except windows */
#अगर !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#पूर्ण_अगर

/* Includes and defines for Hexagon and all HLOS except Windows. */
#अगर !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#पूर्ण_अगर

#अगर !defined(_WIN32)
#    pragma weak remote_system_request
#पूर्ण_अगर

#अगर_घोषित _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#अन्यथा
#     define DSPQUEUE_TIMEOUT 1000000
#पूर्ण_अगर

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API पूर्णांक htpdrv_init(व्योम);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(पूर्णांक domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API पूर्णांक htpdrv_get_arch(पूर्णांक domain, पूर्णांक * arch);

HTPDRV_API पूर्णांक remote_system_request(system_req_payload * req);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
