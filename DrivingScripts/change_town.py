import sys
import carla

client = carla.Client('127.0.0.1', 2000)
num = '2'
if len(sys.argv) > 1:
    num = str(sys.argv[1])
town = 'Town0' + num
client.load_world(town)
