#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

int bind(int descriptor, const struct sockaddr *address, socklen_t length) {
    static int (*original_bind)(int, const struct sockaddr *, socklen_t);
    if (!original_bind) {
        original_bind = dlsym(RTLD_NEXT, "bind");
        if (!original_bind) {
            errno = ENOSYS;
            return -1;
        }
    }
    const char *port_text = getenv("DANZERO_SERVER_PORT");
    if (port_text && address && address->sa_family == AF_INET &&
        length >= sizeof(struct sockaddr_in)) {
        struct sockaddr_in mapped;
        memcpy(&mapped, address, sizeof(mapped));
        if (ntohs(mapped.sin_port) == 23456) {
            char *end;
            long port = strtol(port_text, &end, 10);
            if (*end || port < 1 || port > 65535) {
                errno = EINVAL;
                return -1;
            }
            mapped.sin_port = htons((unsigned short)port);
            return original_bind(descriptor, (struct sockaddr *)&mapped, sizeof(mapped));
        }
    }
    if (port_text && address && address->sa_family == AF_INET6 &&
        length >= sizeof(struct sockaddr_in6)) {
        struct sockaddr_in6 mapped;
        memcpy(&mapped, address, sizeof(mapped));
        if (ntohs(mapped.sin6_port) == 23456) {
            char *end;
            long port = strtol(port_text, &end, 10);
            if (*end || port < 1 || port > 65535) {
                errno = EINVAL;
                return -1;
            }
            mapped.sin6_port = htons((unsigned short)port);
            return original_bind(descriptor, (struct sockaddr *)&mapped, sizeof(mapped));
        }
    }
    return original_bind(descriptor, address, length);
}

int connect(int descriptor, const struct sockaddr *address, socklen_t length) {
    static int (*original_connect)(int, const struct sockaddr *, socklen_t);
    if (!original_connect) {
        original_connect = dlsym(RTLD_NEXT, "connect");
        if (!original_connect) {
            errno = ENOSYS;
            return -1;
        }
    }
    const char *port_text = getenv("DANZERO_SERVER_PORT");
    if (port_text && address && address->sa_family == AF_INET &&
        length >= sizeof(struct sockaddr_in)) {
        struct sockaddr_in mapped;
        memcpy(&mapped, address, sizeof(mapped));
        if (ntohs(mapped.sin_port) == 23456) {
            char *end;
            long port = strtol(port_text, &end, 10);
            if (*end || port < 1 || port > 65535) {
                errno = EINVAL;
                return -1;
            }
            mapped.sin_port = htons((unsigned short)port);
            return original_connect(descriptor, (struct sockaddr *)&mapped, sizeof(mapped));
        }
    }
    return original_connect(descriptor, address, length);
}
