# Complete Auth for Linux

  ⚠️ Not finished yet, the only way to configure is editing scripts.

  ⚠️ As every face recognition tools, its precision is not perfect, recommended similarity threshold is more than 0.6 (I use 0.7 by default and the similarity indice with my friend on my bad HP laptop camera is between 0 and 0.5)

  ⚠️ I made it with my use in mind, and the tool will always work better on intel processors due to openvimo use. It should work on AMD cpus too, but I'm not sure.

## Installation

- First, run download_models.py to download the needed models fore the face identification to work.
- Then, run compile.sh to build and install the PAM module on the system (don't worry, this module is not enabled in PAM config by default, and don't do anything if the daemon doesn't run.)
- Create the host directory: `sudo mkdir /usr/share/face_recognition`
  You can use another path, but you'll need to edit the system module for on-boot autostart.
- Copy the content of the downloaded repo to the new system directory: `sudo cp ./* /usr/share/face_recognition` if you are in the github dir.
- Move system service and dbus config system-wide: `sudo mv /usr/share/face_recognition/etc /`

## Use

- The daemon must be ran as root. You can enable the provided service file for on-boot startup.
- add the line:
  ```
  auth sufficient pam_face.so
  ```
  At the beginning (or the end) of the PAM configs files that you want to allow face recognition.
- To add a new face fo current user
  ```
  /usr/share/face_recognition/facerec.py add [capture path] [face name]
  ```
- To check if the user who is currently facing the computer is the logged in user:
  ```
  /usr/share/face_recognition/facerec.py check
  ```
- To remove all faces for the logged user:
  ```
  /usr/share/face_recognition/facerec.py remove [all]
  ```
- To remove given face names for the user:
  ```
  /usr/share/face_recognition/facerec.py remove (face1) (face2) ...
  ```
## Features:
- Ultra-fast recognition: Utilizes OpenVINO for rapid face detection and identification.

- Low resource usage: Optimized for minimal CPU and memory consumption.

- Camera fallback: Automatically switches between cameras following the configured fallback order.

- Image quality checks: Filters out blurry, under-lit, or over-lit images before recognition.

- Anti-Spoof: Use accurate miniFASnetv2 anti-spoofing algorithm.

- Multi-user support: Stores separate embedding dictionaries per Linux user, allowing multiple profiles.
  
