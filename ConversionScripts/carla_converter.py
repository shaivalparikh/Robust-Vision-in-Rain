
# Based on the default labels for CityScapes; probably inefficient as is
def carla2cityscapes(pixel):
    if pixel == 0:
        return 255
    elif pixel == 1:
        return 2
    elif pixel == 2:
        return 4
    elif pixel == 3:
        return 255
    elif pixel == 4:
        return 11
    elif pixel == 5:
        return 5
    elif pixel == 6:
        return 0
    elif pixel == 7:
        return 0
    elif pixel == 8:
        return 1
    elif pixel == 9:
        return 8
    elif pixel == 10:
        return 13
    elif pixel == 11:
        return 3
    elif pixel == 12:
        return 7
    elif pixel == 13:
        return 10
    elif pixel == 14:
        return 255
    elif pixel == 15:
        return 255
    elif pixel == 16:
        return 255
    elif pixel == 17:
        return 255
    elif pixel == 18:
        return 6
    elif pixel == 19:
        return 255
    elif pixel == 20:
        return 255
    elif pixel == 21:
        return 9
    elif pixel == 22:
        return 9
    else:
        return 255

