#!/bin/sh
# auto compile PAM Module and install it
gcc -fPIC -fno-stack-protector -c pam_face_dbus.c $(pkg-config --cflags glib-2.0 gio-2.0)
ld -x --shared -o pam_face_dbus.so pam_face_dbus.o $(pkg-config --libs glib-2.0 gio-2.0)
sudo cp pam_face_dbus.so /lib/security/
