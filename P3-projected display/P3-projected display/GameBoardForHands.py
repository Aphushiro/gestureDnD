import random
import arcade
import numpy as np
import datetime
import pyglet
import RunFromOtherPy
import ScreenMorphP3
import time

playerMove = 30
playerAttack = 5
divisor = 40 #Higher number = lower grid density
warpBoxScaling = 6
aspectRatio = (3*warpBoxScaling, 3*warpBoxScaling)

# Class containing every value that needs to be saved in between states
class CV_settings:
    def __init__(self):
        self.state = 0
        self.roi = ScreenMorphP3.RoiHandler()
        self.tokenCenters = []
        self.selectionPos = (0, 0)
        self.bgMean = []
    def SetTokenCenters(self, centers):
        self.tokenCenters = centers

    def SetBgMean(self, meanImg):
        self.bgMean = meanImg

    def SetSelection(self, position):
        self.selectionPos = position

    def SetRoi(self, region):
        self.roi = region

    def SetState (self, newState):
        self.state = newState

    def GetSettings(self):
        line = f"Current State: {self.state}\n" \
               f"Roi points:    {self.roi.points}\n" \
               f"Selection pos: {self.selectionPos} \n" \
               f"Token centers: {self.tokenCenters}"
        return line
boardSettings = CV_settings()
# Debug area
#boardSettings.roi.points = ([(318, 67), (658, 91), (663, 432), (291, 416)])

# Run open-cv methods depending on 'state'
def RunOpenCv ():
    print("Running RunOpenCv")
    arcade.start_render()
    DrawGrid(2, False)

    useScreen = pyglet.canvas.get_display()
    useScreen = useScreen.get_screens()[1]
    if len(boardSettings.roi.points) < 4:
        print("Region of interest not defined")
        return
    if boardSettings.state == 0:
        newState, newBgMean, tokenCenters = RunFromOtherPy.StateZero(boardSettings.roi)
        boardSettings.SetState(newState)
        boardSettings.SetBgMean(newBgMean)
        boardSettings.SetTokenCenters(tokenCenters)
        return
    elif boardSettings.state == 1:
        newSelection, nextState, newBgMean, newTokenCenters = RunFromOtherPy.StateOne(boardSettings.roi, boardSettings.bgMean, boardSettings.tokenCenters)
        boardSettings.SetSelection(newSelection)
        boardSettings.SetState(nextState)
        boardSettings.SetBgMean(newBgMean)
        boardSettings.SetTokenCenters(newTokenCenters)
        if nextState == 2:
            Select(boardSettings.selectionPos[0], boardSettings.selectionPos[1])
            DrawGrid(2, False)
        return
    elif boardSettings.state == 2:
        nextState, shouldDraw = RunFromOtherPy.StateTwo(boardSettings.bgMean)  # Set by reading features
        boardSettings.SetState(nextState)
        if not shouldDraw:
            return
        if nextState == 1:
            arcade.start_render()
            DrawGrid(2, True)
            return
        elif nextState == 2:
            HelpMove(boardSettings.selectionPos[0], boardSettings.selectionPos[1])
            DrawGrid(2, False)
        elif nextState == 3:
            HelpAttack(boardSettings.selectionPos[0], boardSettings.selectionPos[1])
            DrawGrid(2, False)
        return
    elif boardSettings.state == 3:
        nextState = RunFromOtherPy.StateThree(boardSettings.bgMean)  # Set by reading features
        boardSettings.SetState(nextState)
        if nextState == 1:
            arcade.start_render()
            DrawGrid(2, True)
            return
        elif nextState == 2:
            DrawRoll()

    else:
        return

# Take a coordinate, and returns the center coordinates of whichever tile the coordinate is within
def FindPosition(x, y):
    #Find Area
    xPos = x-x%divisor+np.ceil(divisor/2).astype(int)
    #print(xPos)
    yPos = y-y%divisor+np.ceil(divisor/2).astype(int)
    #print (yPos)
    pos = (xPos, yPos)
    return [pos[0], pos[1]]

# Finds new orego basaed on aspect ratio and play area size
def GetNewCenter ():
    x, y = arcade.get_window().get_size()
    x, y = FindPosition(x//2, y//2)
    x -= divisor//2 + divisor*aspectRatio[0]//2
    y -= divisor//2 + divisor*aspectRatio[1]//2

    return (x, y)

# Returns properties of the window and grid size
def GetGridProperties ():
    size = arcade.get_window().get_size() # (screen width, screen height)
    xStep = np.ceil(size[0]/divisor).astype(int)
    yStep = np.ceil(size[1]/divisor).astype(int)
    propertyString = f"Grid size: {xStep, yStep}, {xStep*(divisor-1)}x{yStep*(divisor-1)}"
    return propertyString

# Draws the grid, orego marker and square surrounding play area
# Square colour depends on 'isCancelling' parameter
def DrawGrid (gw, isCancelling):
    arcade.set_background_color((0, 0, 0))
    outLineColour = arcade.color.BLACK
    if isCancelling:
        outLineColour = arcade.color.RED

    size = arcade.get_window().get_size() # (screen width, screen height)

    #Drawing the grid
    xStep = np.ceil(size[0]/divisor).astype(int)
    for x in range(0, xStep):
        arcade.draw_line(x*divisor, 0, x*divisor, size[1], arcade.color.WHITE, gw)

    yStep = np.ceil(size[1]/divisor).astype(int)
    for y in range(0, yStep):
        arcade.draw_line(0, y*divisor, size[0], y*divisor, arcade.color.WHITE, gw)

    #Drawing the warp area
    middle = FindPosition(size[0]//2, size[1]//2)

    arcade.draw_rectangle_outline(middle[0] - (divisor // 2), middle[1] - (divisor//2),
                                  aspectRatio[0] * divisor, aspectRatio[1] * divisor, outLineColour, 3)

    # Draw (0, 0)
    x, y = GetNewCenter()
    arcade.draw_circle_outline(x, y, 5, arcade.color.WHITE, 1)

    arcade.finish_render()

# Toggles whether or not edge should contain number, indicating grid index for x and y axes
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

# Drawsa number from 1-20 on screen for two seconds
def DrawRoll ():
    arcade.start_render()
    size = arcade.get_window().get_size()
    roll = str(random.randint(1, 20))
    arcade.draw_text(roll, size[0]//2, size[1]//2, arcade.color.WHITE, 80, 200, align="center")
    arcade.finish_render()
    time.sleep(2)
    DrawGrid(2, False)

# Calculates the positions for each coordinate in the top of a range 'circle' based on range parameter
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

# Draws a line going parallel to the x-axis and another line going parallel to the y-axis
# Both lines ends up at whichever tile is the parameter coordinate x and y is on
def LineupCoords (x, y):
    arcade.start_render()
    pos = FindPosition(x, y)

    arcade.draw_line(pos[0], 0, pos[0], pos[1], arcade.color.DARK_BLUE, 5)
    arcade.draw_line(0, pos[1], pos[0], pos[1], arcade.color.DARK_GREEN, 5)

# Draws a blue range indicator using the CalculateCircle() method
# as well as a coordinate originating from a OpenCV perspective
def HelpMove(x, y):
    a = datetime.datetime.now()
    arcade.start_render()

    scale = 0.48
    x *= -scale
    y *= scale
    size = arcade.get_window().get_size()
    pos = FindPosition(x, y)
    orego = GetNewCenter()
    pos[0] += size[0] - orego[0]
    pos[1] += orego[1]

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

# Draws a square on whichever tile the coordinate (x, y) is located on
def Select (x, y):
    arcade.start_render()

    scale = 0.48
    x *= -scale
    y *= scale
    size = arcade.get_window().get_size()
    pos = FindPosition(x, y)
    orego = GetNewCenter()
    pos[0] += size[0] - orego[0]
    pos[1] += orego[1]

    arcade.draw_rectangle_filled(pos[0], pos[1], divisor, divisor, arcade.color.YELLOW)
    #arcade.draw_rectangle_filled(pos[0]+divisor, pos[1], divisor, divisor, arcade.color.DARK_GREEN)
    #arcade.draw_rectangle_filled(pos[0], pos[1]-divisor, divisor, divisor, arcade.color.DARK_GREEN)
    #arcade.draw_rectangle_filled(pos[0], pos[1]+divisor, divisor, divisor, arcade.color.DARK_GREEN)

# Draws a red range indicator using the CalculateCircle() method
# as well as a coordinate originating from a OpenCV perspective
def HelpAttack (x, y):
    a = datetime.datetime.now()
    arcade.start_render()

    scale = 0.48
    x *= -scale
    y *= scale
    size = arcade.get_window().get_size()
    pos = FindPosition(x, y)
    orego = GetNewCenter()
    pos[0] += size[0] - orego[0]
    pos[1] += orego[1]

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

# The main window, responsible for listening to user input
class MainGame (arcade.Window):
    def __init__(self):
        super().__init__(1326, 780, "Virtual board", fullscreen=False)
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

    # Built-in method, listens to key press events
    def on_key_press(self, key: int, modifiers: int):
        # Fullscreen on keypress
        if key == arcade.key.F:
            useScreen = pyglet.canvas.get_display()
            #width, height = self.get_size()
            #self.set_location(width//4, height//4)

            try:
                useScreen = useScreen.get_screens()[1]
                print(useScreen)
            except:
                print("No second monitor")
                useScreen = useScreen.get_screens()[0]
            # User hits f. Flip between full and not full screen.
            self.set_fullscreen(not self.fullscreen, screen=useScreen)
            print(GetGridProperties())

            # Get the window coordinates. Match viewport to window coordinates
            # so there is a one-to-one mapping.
            width, height = self.get_size()
            self.set_viewport(0, width, 0, height)
        arcade.start_render()
        #posthingWorkPls = (622, 872)
        #Select(posthingWorkPls[0], posthingWorkPls[1])
        #HelpMove(posthingWorkPls[0], posthingWorkPls[1])

        DrawGrid(self.gridLineWidth, self.drawCoords)

        # Toggle whether or not grid should draw coordinate numbers
        if key == arcade.key.T:
            self.drawCoords = not self.drawCoords
        if key == arcade.key.N:
            boardSettings.SetRoi(ScreenMorphP3.DefineRegionOfInterest())
            boardSettings.SetState(0)
        if key == arcade.key.P:
            print(boardSettings.GetSettings())
        if key == arcade.key.SPACE:
            RunOpenCv()
            print(boardSettings.GetSettings())
            print(GetGridProperties())
        if key == arcade.key.B:
            arcade.start_render()
            arcade.finish_render()

game = MainGame()
arcade.run()