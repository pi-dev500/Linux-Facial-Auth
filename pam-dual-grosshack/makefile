build-arch:
	arch-meson . build
	meson compile -C build

install:
	sudo cp ./build/pam/pam_fprintd_grosshack.so /lib/security/pam_dual_grosshack
