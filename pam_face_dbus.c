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
#include <stdlib.h>  // for malloc
#define SOCKET_PATH_FORMAT "/run/user/%d/face.sock"

int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                        int argc, const char **argv) {
    char socket_path[108];
    struct sockaddr_un addr;
    int fd;
    char buf[32];
    ssize_t n;
    uid_t uid = getuid();

    // 🧠 Get username
    const char *username = NULL;
    if (pam_get_user(pamh, &username, NULL) != PAM_SUCCESS || username == NULL) {
        pam_syslog(pamh, LOG_ERR, "Failed to get username");
        return PAM_AUTH_ERR;
    }

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

    // 📦 Build message: "CheckFace\n<username>\n"
    size_t msg_len = strlen("CheckFace\n") + strlen(username) + 2;
    char *msg = malloc(msg_len);
    if (!msg) {
        pam_syslog(pamh, LOG_ERR, "malloc() failed");
        close(fd);
        return PAM_AUTH_ERR;
    }
    snprintf(msg, msg_len, "CheckFace\n%s\n", username);

    write(fd, msg, strlen(msg));
    free(msg);

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
