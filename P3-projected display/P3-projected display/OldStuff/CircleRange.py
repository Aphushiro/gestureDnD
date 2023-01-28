import arcade
playerMove = 60
playerAttack = 20

def Select (x, y):
    arcade.start_render()
    arcade.draw_circle_filled(x, y, 10, arcade.color.APPLE_GREEN)
    arcade.finish_render()

def HelpMove(x, y):
    arcade.start_render()
    arcade.draw_circle_filled(x, y, playerMove, arcade.color.BLUEBERRY)
    arcade.finish_render()

def HelpAttack (x, y):
    arcade.start_render()
    arcade.draw_circle_filled(x, y, playerAttack, arcade.color.ROSE)
    arcade.finish_render()

def Clear ():
    arcade.start_render()
    arcade.finish_render()

class MainGame (arcade.Window):
    def __init__(self):
        super().__init__(1800, 1000, "Virtual board")
        self.x = 0
        self.y = 0
        self.setting = 0

    def on_mouse_press(self, x, y, button, modifier):
        self.x = x
        self.y = y
        if self.setting == 0:
            Clear()
        if self.setting == 1:
            Select(x, y)
        if self.setting == 2:
            HelpMove(x, y)
        if self.setting == 3:
            HelpAttack(x, y)

    def on_key_press(self, key: int, modifiers: int):
        if key == arcade.key.KEY_1:
            self.setting = 0
        if key == arcade.key.KEY_2:
            self.setting = 1
        if key == arcade.key.KEY_3:
            self.setting = 2
        if key == arcade.key.KEY_4:
            self.setting = 3

MainGame()
arcade.run()