#To build and Run Docker Container
docker-compose up && docker build

websocat ws://localhost:8000/ws/publish/{channel_name}/{publisher_name}
websocat ws://localhost:8000/ws/subscribe/{channel_name}/{subscriber_name} 

#for windows
Download websocat.exe, put it in your PATH, and run: