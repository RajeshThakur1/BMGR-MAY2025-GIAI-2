from cartesia import Cartesia

client = Cartesia(api_key="sk_car_X8uxozz9S1q2GJLS3GnfsR")
voices = client.voices.list()
for voice in voices:
    print(voice.id, voice.name)