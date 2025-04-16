// pam_face_socket.c
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <syslog.h>
#include <errno.h>

#define SOCKET_PATH_FORMAT "/run/user/%d/face.sock"

int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                        int argc, const char **argv) {
    char socket_path[108];
    struct sockaddr_un addr;
    int fd;
    char buf[32];
    ssize_t n;
    uid_t uid = getuid();

    snprintf(socket_path, sizeof(socket_path), SOCKET_PATH_FORMAT, uid);

    if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        pam_syslog(pamh, LOG_ERR, "socket() failed: %s", strerror(errno));
        return PAM_AUTH_ERR;
    }

    memset(&addr, 0, sizeof(struct sockaddr_un));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) == -1) {
        pam_syslog(pamh, LOG_ERR, "connect() failed: %s", strerror(errno));
        close(fd);
        return PAM_AUTH_ERR;
    }

    write(fd, "CheckFace", 9);
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (n <= 0) {
        pam_syslog(pamh, LOG_ERR, "read() failed");
        return PAM_AUTH_ERR;
    }

    buf[n] = '\0';
    if (strcmp(buf, "pass") == 0) {
        return PAM_SUCCESS;
    } else {
        pam_syslog(pamh, LOG_NOTICE, "Face recognition failed (%s)", buf);
        return PAM_AUTH_ERR;
    }
}

int pam_sm_setcred(pam_handle_t *pamh, int flags,
                   int argc, const char **argv) {
    return PAM_SUCCESS;
}

