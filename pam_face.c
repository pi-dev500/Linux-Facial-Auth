/* pam_face_dbus.c - PAM module using D-Bus face recognition service */

#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <gio/gio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>

#define DBUS_BUS_NAME      "org.linux.Face"
#define DBUS_OBJECT_PATH   "/org/linux/Face"
#define DBUS_INTERFACE     "org.linux.FaceRecognition"
#define DBUS_METHOD        "CheckFace"
#define DEFAULT_ATTEMPTS   5

/*
 * pam_sm_authenticate: Authenticate user by calling D-Bus face recognition
 */
PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv)
{
    const char *username = NULL;
    int pam_ret;
    GError *error = NULL;
    GDBusConnection *connection = NULL;
    GVariant *reply = NULL;
    gchar *result = NULL;
    guint attempts = DEFAULT_ATTEMPTS;

    /* Retrieve the PAM username */
    pam_ret = pam_get_user(pamh, &username, NULL);
    if (pam_ret != PAM_SUCCESS || username == NULL) {
        pam_syslog(pamh, LOG_ERR, "Failed to get username");
        return PAM_AUTH_ERR;
    }

    /* Connect to the system bus */
    connection = g_bus_get_sync(G_BUS_TYPE_SYSTEM, NULL, &error);
    if (!connection) {
        pam_syslog(pamh, LOG_ERR, "Failed to connect to system bus: %s", error->message);
        g_clear_error(&error);
        return PAM_AUTH_ERR;
    }

    /* Call the CheckFace method */
    reply = g_dbus_connection_call_sync(
        connection,
        DBUS_BUS_NAME,        /* bus name */
        DBUS_OBJECT_PATH,     /* object path */
        DBUS_INTERFACE,       /* interface name */
        DBUS_METHOD,          /* method name */
        g_variant_new("(si)", username, attempts), /* input signature: s = string, i = int */
        G_VARIANT_TYPE("(s)"), /* expected output: single string */
        G_DBUS_CALL_FLAGS_NONE,
        -1,                   /* default timeout */
        NULL,                 /* cancellable */
        &error);

    if (!reply) {
        pam_syslog(pamh, LOG_ERR, "D-Bus call failed: %s", error->message);
        g_clear_error(&error);
        g_object_unref(connection);
        return PAM_AUTH_ERR;
    }

    /* Parse the response */
    g_variant_get(reply, "(s)", &result);
    g_variant_unref(reply);
    g_object_unref(connection);

    /* Evaluate result */
    if (g_strcmp0(result, "pass") == 0) {
        g_free(result);
        return PAM_SUCCESS;
    } else {
        pam_syslog(pamh, LOG_NOTICE, "Face recognition failed: %s", result);
        g_free(result);
        return PAM_AUTH_ERR;
    }
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
    int argc, const char **argv) {
return PAM_SUCCESS;
}