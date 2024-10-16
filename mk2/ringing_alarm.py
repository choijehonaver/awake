import pygame

def select_alarm(result) :
    if result == 0:
        # sound_alarm("power_alarm.wav")
        result_message = "Power Nap Detected!"
    elif result == 1 :
        # sound_alarm("nomal_alarm.wav")
        result_message = "Normal Nap Detected!"
    else :
        # sound_alarm("short_alarm.mp3")
        result_message = "Short Nap Detected!"

    # sound_alarm(alarm_sound)
    print(result_message)
    return result_message 

def sound_alarm(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    

