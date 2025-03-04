
import asyncio

from abc import ABC, abstractmethod
from dataclasses import dataclass
from bleak import BleakClient

COMMAND_CHARACTERISTIC_UUID = "eee4879a-80a8-4d33-af74-b05826ee658f"


class DifferentialDriveRobot(ABC):
    @dataclass
    class Command:
        left_target_velocity_rad: float
        right_target_velocity_rad: float

    @abstractmethod
    def execute_command(command: Command):
        """Execute a single command."""


class MockRobot(DifferentialDriveRobot):
    def execute_command(self, _command):
        pass

class BluetoothLowEnergyRobot(DifferentialDriveRobot):
    def __init__(
        self,
        address,
    ):
        self.__event_loop = asyncio.get_event_loop()
        self.__client = BleakClient(address)
        self.__event_loop.run_until_complete(self.__client.connect())

    def __del__(self):
        self.__event_loop.run_until_complete(self.__client.disconnect())

    def execute_command(self, command: DifferentialDriveRobot.Command):
        message_out = f"{command.left_target_velocity_rad:.3f} {command.right_target_velocity_rad:.3f}"
        self.__event_loop.run_until_complete(
            self.__client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, bytes(message_out, encoding="utf8"))
        )
