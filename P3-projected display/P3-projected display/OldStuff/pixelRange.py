import random
import time

import arcade
import numpy as np
import datetime
import pyglet
playerMove = 30
playerAttack = 5
divisor = 40 #Higher number = lower grid density

def DrawGrid (gw, coordsOn):
    arcade.set_background_color((0, 0, 0))
    if coordsOn:
        ToggleText()
    size = arcade.get_window().get_size() # (screen width, screen height)

    xStep = np.ceil(size[0]/divisor).astype(int)
    for x in range(0, xStep):
        arcade.draw_line(x*divisor, 0, x*divisor, size[1], arcade.color.WHITE, gw)

    yStep = np.ceil(size[1]/divisor).astype(int)
    for y in range(0, yStep):
        arcade.draw_line(0, y*divisor, size[0], y*divisor, arcade.color.WHITE, gw)

    arcade.finish_render()

    print(f"Grid size: {xStep, yStep}, {xStep*(divisor-1)}x{yStep*(divisor-1)}")

def ToggleText ():
    size = arcade.get_window().get_size()  # (screen width, screen height)

    xStep = size[0]//divisor
    for x in range(1, xStep):
        pos = FindPosition(x*divisor, divisor//2)
        arcade.draw_text(f"{x}", pos[0], pos[1], arcade.color.WHITE)
        print(f"({x}, 0) at pos ({pos[0]}, {pos[1]})")

    yStep = size[1]//divisor
    for y in range(1, yStep):
        pos = FindPosition(divisor//2, y*divisor)
        arcade.draw_text(f"{y}", pos[0], pos[1], arcade.color.WHITE)

def FindPosition(x, y):
    #Find Area
    xPos = x-x%divisor+np.ceil(divisor/2).astype(int)
    #print(xPos)
    yPos = y-y%divisor+np.ceil(divisor/2).astype(int)
    #print (yPos)
    pos = (xPos, yPos)
    return pos

def CalculateCircle (steps):
    coordinates = []
    coordinates.append([1, steps])
    coordinates.append(([steps, 1]))
    y = 1  # starts at 1
    x = steps
    #Magical dnd algorithm that I will never touch again
    #aka. calculating coordinates for 1/4 shell
    for i in range(steps-2):
        print(f"({steps-i},{y})")
        if y+2 < steps-i:
            y = y+2
            x = x-1
            coordinates.append([x, y])

        elif y+2 == steps-i:
            y = y+1
            x = x-1
            coordinates.append([x, y])
        elif y+1 == steps-i:
            y = y+1
            x = x-1
            coordinates.append([x, y])
        elif y + 1 > steps-i and x%2 == 1:
            x = x-1
            coordinates.append([x, y])
        else:
            y = y+1
            x = x-1
            coordinates.append([x, y])
    flip = []
    for cord in coordinates:
        flip.append([-cord[0], cord[1]])
    for cord in flip:
        coordinates.append(cord)
    coordinates.append([0, steps])

    print(coordinates)
    return coordinates

def LineupCoords (x, y):
    arcade.start_render()
    pos = FindPosition(x, y)

    arcade.draw_line(pos[0], 0, pos[0], pos[1], arcade.color.DARK_BLUE, 5)
    arcade.draw_line(0, pos[1], pos[0], pos[1], arcade.color.DARK_GREEN, 5)

def HelpMove(x, y):
    a = datetime.datetime.now()
    arcade.start_render()
    pos = FindPosition(x, y)
    steps = np.ceil(playerMove/5).astype(int)

    coordinates = CalculateCircle(steps)
    for cords in coordinates:
        startEndX = pos[0]+(cords[0]*divisor)
        startY = pos[1]+(cords[1]*divisor)+divisor/2
        endY = pos[1]+(-cords[1]*divisor)-divisor/2
        arcade.draw_line(startEndX, startY, startEndX, endY, arcade.color.DARK_BLUE, divisor)
    b = datetime.datetime.now()
    delta = (b-a)
    delta = int(delta.total_seconds()*1000)
    print(f"Delta: {delta}")

def Select (x, y):
    arcade.start_render()
    pos = FindPosition(x, y)
    arcade.draw_rectangle_filled(pos[0], pos[1], divisor, divisor, arcade.color.GREEN)

def DrawRoll ():
    arcade.start_render()
    size = arcade.get_window().get_size()
    roll = str(random.randint(1, 20))
    arcade.draw_text(roll, size[0]//2, size[1]//2, arcade.color.WHITE, 80, 200, align="center")
    arcade.finish_render()
    time.sleep(2)
    DrawGrid(2, False)

def HelpAttack (x, y):
    a = datetime.datetime.now()
    arcade.start_render()
    pos = FindPosition(x, y)
    steps = np.ceil(playerAttack / 5).astype(int)
    coordinates = CalculateCircle(steps)
    for cords in coordinates:
        startEndX = pos[0]+(cords[0]*divisor)
        startY = pos[1]+(cords[1]*divisor)+divisor/2
        endY = pos[1]+(-cords[1]*divisor)-divisor/2
        arcade.draw_line(startEndX, startY, startEndX, endY, arcade.color.DARK_GREEN, divisor)
    b = datetime.datetime.now()
    delta = (b-a)
    delta = int(delta.total_seconds()*1000)
    print(f"Delta: {delta}")

class MainGame (arcade.Window):
    def __init__(self):
        super().__init__(800, 800, "Virtual board", fullscreen=False)
        self.x = 0
        self.y = 0
        self.setting = 0
        self.gridLineWidth = 2
        self.drawCoords = False

        if self.gridLineWidth >= divisor/2:
            self.gridLineWidth = 2
            print("Rescaled gridLineWidth to be = 2")

        arcade.start_render()
        DrawGrid(self.gridLineWidth, self.drawCoords)

    def on_mouse_press(self, x, y, button, modifier):
        self.x = x
        self.y = y
        if self.setting == 0:
            arcade.start_render()
            DrawGrid(self.gridLineWidth, self.drawCoords)
        if self.setting == 1:
            Select(x, y)
            DrawGrid(self.gridLineWidth, self.drawCoords)
        if self.setting == 2:
            HelpMove(x, y)
            DrawGrid(self.gridLineWidth, self.drawCoords)
        if self.setting == 3:
            HelpAttack(x, y)
            DrawGrid(self.gridLineWidth, self.drawCoords)
        if self.setting == 4:
            LineupCoords(x, y)
            DrawGrid(self.gridLineWidth, self.drawCoords)
        if self.setting == 5:
            DrawRoll()

    def on_key_press(self, key: int, modifiers: int):
        if key == arcade.key.F:
            useScreen = pyglet.canvas.get_display()
            try:
                useScreen = useScreen.get_screens()[1]
                print(useScreen)
            except:
                print("No second monitor")
                useScreen = useScreen.get_screens()[0]
            # User hits f. Flip between full and not full screen.
            self.set_fullscreen(not self.fullscreen, screen=useScreen)
            print(f"Size: {useScreen.width}X, {useScreen.height}Y")

            # Get the window coordinates. Match viewport to window coordinates
            # so there is a one-to-one mapping.
            width, height = self.get_size()
            self.set_viewport(0, width, 0, height)

        if key == arcade.key.T:
            self.drawCoords = not self.drawCoords
        if key == arcade.key.KEY_1:
            self.setting = 0
        if key == arcade.key.KEY_2:
            self.setting = 1
        if key == arcade.key.KEY_3:
            self.setting = 2
        if key == arcade.key.KEY_4:
            self.setting = 3
        if key == arcade.key.KEY_5:
            self.setting = 4
        if key == arcade.key.KEY_6:
            self.setting = 5

game = MainGame()
arcade.run()