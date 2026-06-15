import pygame

pygame.init()

# Window
game_w,game_h=800,600
panel_h=120
w,h=game_w,game_h+panel_h
screen=pygame.display.set_mode((w,h))
pygame.display.set_caption("Car Simulator")
font=pygame.font.Font(None,22)






# colors
game_bg=(18,18,18)
panel_bg=(40,40,40)
car_color=(0,200,0)
obst_color=(255,200,200)


# car physics
car_w,car_h=45,20
max_speed=300.0
max_accel=600.0
brake_force=4.0
friction=220.0



class Car:
    def _init__(self):
        self.x=car_w
        self.y=car_h
        self.angle=0.0
        self.speed=0.0


    def draw(self,surf):
        car_s=pygame.Surface((car_w,car_h))
        pygame.draw.rect(car_s,car_color,(0,0,car_w,car_h),border_radius=5)
        

def main():
    clock=pygame.time.Clock()
    car=Car()
    running=True

    while running:
        clock.tick(60)/1000.0
        screen.fill(game_bg)
        pygame.draw.rect(screen,panel_bg,(0,game_h,w,panel_h))

        # events
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False

       
    pygame.quit()

if __name__=="__main__":
    main()

