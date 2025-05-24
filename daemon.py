#!/usr/bin/env/python
# daemon.py
import os
from gi.repository import GLib
from pydbus import SystemBus
from facerec import check
from threading import Thread
from time import monotonic

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
            <method name='GetIdentifier'>
                <arg type='s' name='username' direction='in'/>
                <arg type='i' name='timeout' direction='in'/>
                <arg type='s' name='response' direction='out'/>
            </method>
            <method name='GetState'>
                <arg type='s' name='request_id' direction='in'/>
                <arg type='s' name='response' direction='out'/>
            </method>
            <method name='ReleaseDevice'>
                <arg type='s' name='request_id' direction='in'/>
            </method>
        </interface>
    </node>
    """
    def __init__(self):
        self.checkthread = None
        self.states = {}
        self.commands_trigger = {}

    def CheckFace(self, user, attempts = 5, timeout = 1):
        return check(user, attempts, timeout)

    def GetIdentifier(self, username, timeout = 3):
        """
        Build an identifier and setup class for running.
        """
        id = f'|{username}|{timeout}|{monotonic()}|'
        self.states[id] = None
        self.commands_trigger[id] = []
        if self.checkthread is None: # Start recognition if device is available
            self.checkthread = Thread(target = self.RunRecognition, args = (id,))
            self.checkthread.start()
        return id

    def RunRecognition(self, recid):
        _, username, timeout, treq, _ = recid.split("|")
        timeout = int(timeout)
        self.states[recid] = check(username=username, n_try=..., timeout=timeout, commands_trigger=self.commands_trigger[recid])
        self.checkthread = None # no more need to keep the thread as it ended
        del self.commands_trigger[recid]
    
    def GetState(self, request_id):
        if request_id in self.states and self.states[request_id] is not None:
            return self.states[request_id]
        elif self.checkthread is None:
            self.checkthread = Thread(target = self.RunRecognition, args = (request_id,))
            self.checkthread.start()
        return "None" # if first conditions is not respected, there is no state

    def ReleaseDevice(self,request_id):
        if self.checkthread is not None and request_id in self.commands_trigger:
            self.commands_trigger[request_id].append(1) # send quit signal
            self.checkthread.join() # wait until the thread completely stop
        del self.states[request_id]
try:
    # D-Bus
    loop = GLib.MainLoop()
    bus = SystemBus()
    bus.publish("org.linux.Face", Daemon())
    print("Daemon running. Ctrl+C to stop.")
    loop.run()
except KeyboardInterrupt:
    print("Stopping...")

