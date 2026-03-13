# import pyorbbecsdk as ob

# def main():
#     # Create context
#     ctx = ob.Context()

#     # Query connected devices
#     device_list = ctx.query_devices()

#     if device_list.get_count() == 0:
#         print("No Orbbec device found")
#         return

#     # Get first device
#     device = device_list.get_device(0)

#     print("Device Information")
#     print("------------------")
#     print("Name:", device.get_device_info().get_name())
#     print("Serial:", device.get_device_info().get_serial_number())
#     print("Firmware:", device.get_device_info().get_firmware_version())
#     print()

#     # Query sensors
#     sensor_list = device.query_sensors()

#     print("Sensors detected:")
#     print("------------------")

#     for i in range(sensor_list.get_count()):
#         sensor = sensor_list.get_sensor(i)
#         sensor_type = sensor.get_type()

#         print("Sensor:", sensor_type)

#         profiles = sensor.get_stream_profile_list()

#         if profiles is not None:
#             for j in range(profiles.get_count()):
#                 profile = profiles.get_stream_profile(j)

#                 print(
#                     "  Resolution:",
#                     profile.get_width(),
#                     "x",
#                     profile.get_height(),
#                     "FPS:",
#                     profile.get_fps()
#                 )

#         print()

# if __name__ == "__main__":
#     main()


import usb.core
import usb.util

def list_usb_devices():
    devices = usb.core.find(find_all=True)

    for dev in devices:
        try:
            manufacturer = usb.util.get_string(dev, dev.iManufacturer)
            product = usb.util.get_string(dev, dev.iProduct)
        except:
            manufacturer = None
            product = None

        if manufacturer and "orbbec" in manufacturer.lower() or \
           product and "orbbec" in product.lower() or \
           product and "dabai" in product.lower():

            print("Device Found")
            print("------------")
            print("Manufacturer:", manufacturer)
            print("Product:", product)
            print("Vendor ID:", hex(dev.idVendor))
            print("Product ID:", hex(dev.idProduct))
            print("Bus:", dev.bus)
            print("Address:", dev.address)
            print()

            for cfg in dev:
                print("Configuration:", cfg.bConfigurationValue)
                for intf in cfg:
                    print(" Interface:", intf.bInterfaceNumber)
                    print("  Class:", intf.bInterfaceClass)
                    print("  Subclass:", intf.bInterfaceSubClass)

            print("\n")

if __name__ == "__main__":
    list_usb_devices()