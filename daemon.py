#!/usr/bin/env/python
# daemon.py
import os
from gi.repository import GLib
from pydbus import SystemBus
from facerec import check


# D-Bus Daemon
class Daemon(object):
    """
    <node>
        <interface name='org.linux.FaceRecognition'>
            <method name='CheckFace'>
                <arg type='s' name='user' direction='in'/>
                <arg type='i' name='attempts' direction='in'/>
                <arg type='s' name='response' direction='out'/>
            </method>
        </interface>
    </node>
    """
    def CheckFace(self, user,attempts=5):
        return check(user,attempts)
try:
    # D-Bus
    loop = GLib.MainLoop()
    bus = SystemBus()
    bus.publish("org.linux.Face", Daemon())
    print("Daemon running. Ctrl+C to stop.")
    loop.run()
except KeyboardInterrupt:
    print("Stopping...")

