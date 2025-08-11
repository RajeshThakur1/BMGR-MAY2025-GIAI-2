# 🤖 Voice Bot - Interactive AI Assistant

A multilingual voice AI assistant built with LiveKit Agents, supporting English, Hindi, and Urdu conversations through a beautiful web interface.

## 🌟 Features

- **Push-to-talk interface** - Click and hold to speak
- **Multilingual support** - English, Hindi, Urdu
- **Real-time voice processing** - Fast STT, LLM, and TTS pipeline
- **Beautiful web UI** - Modern, responsive design
- **Live transcription** - See conversations in real-time

## 🚀 Quick Start

### 1. Setup Environment

Make sure you're in the project directory and activate the virtual environment:

```bash
cd voice_bot
source venv/bin/activate
```

### 2. Install Dependencies (if not already done)

```bash
pip install "livekit-agents[openai,cartesia,silero,turn-detector]~=1.0" "livekit-plugins-noise-cancellation~=0.2" "python-dotenv"
```

### 3. Download Model Files

```bash
python agent.py download-files
```

### 4. Run the Application

#### Option A: Web Interface (Recommended)

**Terminal 1** - Start the web server:
```bash
python server.py
```

**Terminal 2** - Start the voice agent:
```bash
python web_agent.py dev
```

Then open your browser and go to: **http://localhost:8080/app.html**

#### Option B: Console Mode (Testing)

```bash
python agent.py console
```

#### Option C: LiveKit Playground

```bash
python agent.py dev
```

Then visit the LiveKit Agents playground URL shown in the terminal.

## 📱 How to Use the Web Interface

1. **Open the web interface** at `http://localhost:8080/app.html`
2. **Enter room details**:
   - Room Name: `voice-chat-room` (or any name)
   - Your Name: Enter your name
3. **Click "Connect to Voice Bot"**
4. **Start talking**:
   - Click and hold the microphone button
   - Speak clearly
   - Release when done
5. **Listen to responses** - The AI will respond with voice and text

## 🔧 Configuration

### Environment Variables (.env file)

```env
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
CARTESIA_API_KEY=your_cartesia_key
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_SECRET=your_livekit_secret
LIVEKIT_API_KEY=your_livekit_key
```

### File Structure

```
voice_bot/
├── agent.py          # Original console agent
├── web_agent.py      # Web-optimized agent
├── app.html          # Main web interface
├── index.html        # Demo interface
├── server.py         # Web server
├── test_audio.py     # Audio testing
├── .env              # Environment variables
└── README.md         # This file
```

## 🛠️ Troubleshooting

### Audio Issues
- **No audio output**: Check system volume and browser permissions
- **Microphone not working**: Allow microphone access in browser
- **Connection fails**: Ensure both terminals are running

### API Issues
- **Deepgram 402 error**: Use OpenAI STT instead (already configured)
- **OpenAI rate limits**: Check your API quota
- **Cartesia issues**: Verify API key and voice ID

### Connection Issues
- **Port 8080 in use**: Run `python server.py 8081` for different port
- **LiveKit connection**: Check your LiveKit credentials in .env

## 🎯 Commands Reference

| Command | Purpose |
|---------|---------|
| `python server.py` | Start web interface server |
| `python web_agent.py dev` | Start agent for web connections |
| `python agent.py console` | Test in terminal |
| `python agent.py download-files` | Download required models |
| `python test_audio.py console` | Test simplified audio setup |

## 🌐 Accessing the Interface

- **Main Interface**: http://localhost:8080/app.html
- **Demo Interface**: http://localhost:8080/index.html
- **Server Status**: Check terminal for connection details

## 💡 Tips

- Speak clearly and at normal volume
- Wait for the processing indicator before speaking again
- Try different languages - the bot supports multilingual conversations
- Use the console mode for debugging audio issues

## 🔄 Model Information

The application downloads these models automatically:

- **Turn Detector Models**: For end-of-turn detection (multilingual)
- **Silero VAD Models**: For voice activity detection
- **Noise Cancellation Models**: For enhanced audio quality

Enjoy chatting with your AI voice assistant! 🎉
