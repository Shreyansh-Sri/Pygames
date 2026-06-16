import pygame
import math

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
    def __init__(self):
        self.x=game_w//2
        self.y=game_h//2
        self.angle=0.0
        self.speed=0.0


    def draw(self,surf):
        car_s=pygame.Surface((car_w,car_h))
        pygame.draw.rect(car_s,car_color,(0,0,car_w,car_h),border_radius=5)
        rotated=pygame.transform.rotate(car_s,self.angle)
        rect=rotated.get_rect(center=(self.x,self.y))
        screen.blit(rotated, rect)


    def update(self,dt):
        keys=pygame.key.get_pressed()
        if keys[pygame.K_w]:
               self.speed+=max_accel * dt 
               self.speed=min(self.speed, max_speed) 
        if keys[pygame.K_s]:
                self.speed-=friction*dt
                self.speed=max(self.speed,0.0)
        if keys[pygame.K_a]:
                self.angle+=3.0
        if keys[pygame.K_d]:
                self.angle-=3.0     
        # position update ho rha hai
        self.x +=math.cos(math.radians(self.angle))*self.speed * dt
        self.y -=math.sin(math.radians(self.angle))*self.speed * dt
        self.x=max(0,min(self.x,game_w))
        self.y=max(0,min(self.y,game_h))


def main():
    clock=pygame.time.Clock()
    car=Car()
    running=True

    while running:
        dt=clock.tick(60)/1000.0
        screen.fill(game_bg)
        car.update(dt)
        car.draw(screen)
        pygame.draw.rect(screen,panel_bg,(0,game_h,w,panel_h))

        # events
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False
        pygame.display.update()
       
    pygame.quit()

if __name__=="__main__":
    main()

