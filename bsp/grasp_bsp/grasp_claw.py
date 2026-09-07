import time
import minimalmodbus
import numpy as np
import threading

POSITION_HIGH_8 = 0x0102  # 位置寄存器高八位
POSITION_LOW_8 = 0x0103  # 位置寄存器低八位
SPEED = 0x0104
FORCE = 0x0105
MOTION_TRIGGER = 0x0108

lock = threading.Lock()
class grasp_claw:
    def __init__(self, gripper_port ='COM10', gripper_baudrate=115200, gripper_address=1,
                ):
        
        self.instrument = minimalmodbus.Instrument(gripper_port, gripper_address)
        self.instrument.serial.baudrate = gripper_baudrate
        self.instrument.serial.timeout = 1


# Define the gripper control class
    def write_position_high8(self, value):
        with lock:
            self.instrument.write_register(POSITION_HIGH_8, value, functioncode=6)

    def write_position_low8(self, value):
        with lock:
            self.instrument.write_register(POSITION_LOW_8, value, functioncode=6)

    def write_position(self, value):
        with lock:
            self.instrument.write_long(POSITION_HIGH_8, value)

    def write_speed(self, speed):
        with lock:
            self.instrument.write_register(SPEED, speed, functioncode=6)

    def write_force(self, force):
        with lock:
            self.instrument.write_register(FORCE, force, functioncode=6)

    def trigger_motion(self):
        with lock:
            self.instrument.write_register(MOTION_TRIGGER, 1, functioncode=6)

    def read_position(self):
        with lock:
            high = self.instrument.read_register(POSITION_HIGH_8, functioncode=3)
            low = self.instrument.read_register(POSITION_LOW_8, functioncode=3)
            position = (high << 8) | low
            return position

    def grip(self, position, speed, force):
        self.write_position(position)
        self.write_speed(speed)
        self.write_force(force)
        self.trigger_motion()
        time.sleep(2)
        return self.read_position()
    

