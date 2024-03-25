import numpy as np

class Colour:
    def __init__(self, red, green, blue):
        self.setRed(red)
        self.setGreen(green)
        self.setBlue(blue)

    def setRed(self, red):
        if red < 0:
            self.red = 0
        elif red > 1:
            self.red = 255
        else:
            self.red = int(red * 255)

    def setGreen(self, green):
        if green < 0:
            self.green = 0
        elif green > 1:
            self.green = 255
        else:
            self.green = int(green * 255)

    def setBlue(self, blue):
        if blue < 0:
            self.blue = 0
        elif blue > 1:
            self.blue = 255
        else:
            self.blue = int(blue * 255)

    def getColour(self):
        return np.array([self.red, self.green, self.blue])

    def __sub__(self, other):
        return np.sqrt((self.red - other.red)**2 + (self.green - other.green)**2 + (self.blue - other.blue)**2)
