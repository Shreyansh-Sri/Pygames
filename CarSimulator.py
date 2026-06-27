import pygame
import math
import cv2
import threading
import time

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
# global variable
car_w,car_h=45,20
max_speed=300.0
max_accel=600.0
brake_force=4.0
friction=220.0
# for smooth face rect
smooth_x,smooth_y,smooth_fw,smooth_fh=0,0,0,0
cap=cv2.VideoCapture(0)
eye_command="forward"
lastseencmd="forward"
cmdstarttime=time.time()
lock=threading.Lock()

def camera_thread():
    global eye_command,lastseencmd,cmdstarttime
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while(True):
        ret,frame=cap.read()
        if not ret:
              continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.1,5)


        for(x,y,fw,fh) in faces:
            face_roi=gray[y:y+fh//2,x:x+fw]
            eyes=eye_cascade.detectMultiScale(face_roi,1.1,5)

            cv2.rectangle(frame,(x,y),(x+fw,y+fh),(0,255,0),2)
            for(ex,ey,ew,eh) in eyes:
                    eye_centre=(x+ex+ew//2,y+ey+eh//2)
                    radius=ew//2
                    cv2.circle(frame,eye_centre,radius,(255,255,0),2)
                    cv2.circle(frame,eye_centre,3,(0,0,255),-1)


            # count kar rhe
            if len(eyes)==0:
                cmd="brake"
            elif len(eyes)==1:
                ex,ey,ew,eh=eyes[0]
                eye_centre_x=ex+ew//2
                if eye_centre_x<fw//2:
                     cmd="left"
                else:
                    cmd="right"     
            else:
                cmd="forward"  

                          
            
            if cmd!=lastseencmd:
                lastseencmd=cmd
                cmdstarttime=time.time()
            else:
                if time.time()-cmdstarttime>0.3:
                    with lock:
                        eye_command=cmd
            print (eye_command)          
        if len(faces)==0:
            cmd="brake"
            if cmd!=lastseencmd:
                lastseencmd=cmd
                cmdstarttime=time.time()
            else:
                if time.time()-cmdstarttime>0.3:
                    with lock:
                        eye_command=cmd
            print(eye_command)
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        
class Car:
    def __init__(self):
        self.x=game_w//2
        self.y=game_h//2
        self.angle=0.0
        self.speed=0.0
        self.image=pygame.image.load("car.png")
        self.image=pygame.transform.scale(self.image,(car_w,car_h))


    def draw(self,surf):

        car_s=pygame.Surface((car_w,car_h))
        pygame.draw.rect(car_s,car_color,(0,0,car_w,car_h),border_radius=5)
        # car_s=self.image
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
                if self.speed>0:
                     self.angle+=3.0
        if keys[pygame.K_d]:
            if self.speed>0:
                 self.angle-=3.0
        global eye_command
        with lock:
             cmd=eye_command
        
        if cmd=="forward":
             self.speed+=max_accel*dt
             self.speed=min(self.speed,max_speed)
        elif cmd=="left":
             if self.speed>0:
                  self.angle+=3.0          
        elif cmd=="right":
             if self.speed>0:
                  self.angle-=3.0
        elif cmd=="brake":
             self.speed-=friction*dt
             self.speed=max(self.speed,0.0)          
             
        # Friction add kr rhe
        self.speed-=friction*dt
        self.speed=max(self.speed,0.0)       
        # position update ho rha hai
        self.x +=math.cos(math.radians(self.angle))*self.speed * dt
        self.y -=math.sin(math.radians(self.angle))*self.speed * dt
        self.x=max(0,min(self.x,game_w))
        self.y=max(0,min(self.y,game_h))

def main():
    global eye_command
    clock=pygame.time.Clock()
    car=Car()
    t=threading.Thread(target=camera_thread,daemon=True)
    t.start()
    running=True
    emotion="neutral"

    while running:
        dt=clock.tick(60)/1000.0
        screen.fill(game_bg)
        car.update(dt)
        car.draw(screen)
        pygame.draw.rect(screen,panel_bg,(0,game_h,w,panel_h))
        
        # Panel me changes kar rhe hh
        speed_text=font.render(f"SPEED: {int(car.speed)}",True,(255,255,255))
        angle_text=font.render(f"ANGLE: {int(car.angle% 360)}",True,(255,255,255))
        emotion_text=font.render(f"CMD: {eye_command}",True,(255,255,255))
        
        
        # Panel ko update ho rha
        screen.blit(speed_text,(20,game_h+20))
        screen.blit(angle_text,(220,game_h+20))
        screen.blit(emotion_text,(420,game_h+20)) 

        # events
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False
        pygame.display.update()
          
    pygame.quit()

if __name__=="__main__":
    main()

