import asyncio
import traceback
import pyaudio
import os
import io
from PIL import Image
from gpiozero import Button, LED

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

BUTTON_PIN = 23
LED_PIN = 25

led = LED(LED_PIN)

# --- AUDIO CONFIGURATION ---
# Format: 16-bit integer (standard for voice audio)
FORMAT = pyaudio.paInt16
# Channels: 1 = Mono (Microphone input is usually mono)
CHANNELS = 1
# Sample Rates:
# 16000Hz is standard for Speech-to-Text (sending)
# 24000Hz is the high-quality output from Gemini (receiving)
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000

# *** CRITICAL FIX FOR RASPBERRY PI ***
# The Chunk Size is how much audio we process at once.
# 1024 was too small, causing the CPU to lag (Underrun).
# 4096 gives the Pi more time to process data, stopping the stutter.
CHUNK_SIZE = 4096

MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

client = genai.Client(
    http_options={"api_version": "v1alpha"}, api_key=os.getenv("GOOGLE_API_KEY")
)

CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    system_instruction="""You are a GCSE flashcard revision assistant.
    You are given a photo of a flashcard, and you generate questions based on the card and the topic.
    Give each question a number of marks it's out of.
    Output to the user only the question. You grade the user's answer and then move on to the next question.

    Do not move on to the next question until the user has answered the current one. Do not answer the question for them unless they say give up.
    """,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")
        )
    ),
    proactivity={"proactive_audio": True},
)


def snap():
    led.on()
    print("Taking photo...")
    os.system(
        "libcamera-still --width 1536 --height 1024 -o /home/pi/flashcards/temp/card.jpg -n"
    )
    print("Photo taken!")
    led.off()


def get_image_bytes(file_path):
    # 1. Open the image
    img = Image.open(file_path)

    # 2. Resize it to max 2048 (preserves aspect ratio)
    img.thumbnail((2048, 2048))

    # 3. Create a "fake file" in memory
    img_byte_arr = io.BytesIO()

    # 4. Save the image into that fake file as JPEG
    img.save(img_byte_arr, format="JPEG")

    # 5. Return the raw bytes
    return img_byte_arr.getvalue()


class RestartSession(Exception):
    pass


class AudioLoop:
    def __init__(self):
        # Asyncio Queues act like conveyor belts.
        # They allow one part of the code to write data (like the mic)
        # while another part reads it (sending to wifi) without waiting for each other.
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.audio_stream = None
        self.button = Button(BUTTON_PIN)

        self.ai_is_speaking = False

    # TASK 1: Handle Keyboard Input
    # async def send_text(self):
    #    while True:
    # specifically use to_thread so 'input' doesn't freeze the audio
    #        text = await asyncio.to_thread(input, "message > ")
    #        if text.lower() == "q":
    #            break
    #        if self.session is not None:
    #            # Sends text to Gemini (e.g., if you type "Hello")
    #            await self.session.send_client_content(
    #                turns=types.Content(
    #                   role="user", parts=[types.Part(text=text or ".")]
    #               )
    #           )

    # Send Audio to Gemini
    async def send_realtime(self):
        while True:
            # Wait for audio data from the microphone queue
            msg = await self.out_queue.get()
            if self.session is not None:
                # Send the raw audio bytes to Google
                await self.session.send_realtime_input(
                    media=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )

    # TASK 3: Listen to Microphone
    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()

        # Open the connection to the physical microphone
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )

        # If the buffer overflows, just ignore it (prevents crashing)
        kwargs = {"exception_on_overflow": False}

        while True:
            # Read raw bytes from hardware
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            # Put data onto the conveyor belt for 'send_realtime' to pick up
            if not self.ai_is_speaking and self.audio_in_queue.empty():
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            else:
                pass

    # TASK 4: Receive Audio from Gemini
    async def receive_audio(self):
        while True:
            if self.session is None:
                await asyncio.sleep(0.1)
                continue

            # This listens to the websocket for response chunks
            turn = self.session.receive()
            async for response in turn:
                # If Gemini sends audio data...
                if data := response.data:
                    # Put it on the 'audio_in_queue' conveyor belt
                    self.audio_in_queue.put_nowait(data)
                    continue
                # If Gemini sends text (captions/logs)...
                if text := response.text:
                    print(text, end="")

    # TASK 5: Play Sound on Speakers
    async def play_audio(self):
        # Setup the connection to the speakers
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            # Pick up audio chunks from the 'receive_audio' conveyor belt
            bytestream = await self.audio_in_queue.get()
            # Write them to the hardware speakers
            self.ai_is_speaking = True

            await asyncio.to_thread(
                stream.write, bytestream, exception_on_underflow=False
            )

            # Check if queue is empty to un-mute mic
            if self.audio_in_queue.empty():
                # Small delay to ensure the sound actually finished playing
                await asyncio.sleep(0.2)
                self.ai_is_speaking = False

    # TASK 6: Watch for Button Press
    async def monitor_button(self):
        print(f"Button monitoring active on GPIO {BUTTON_PIN}...")
        while True:
            # Check if button is pressed
            if self.button.is_pressed:
                print("\n!!! Button Pressed: Restarting Session !!!")
                # Add a small delay so we don't trigger it twice instantly
                await asyncio.sleep(1)
                # Raise exception to break the TaskGroup
                raise RestartSession()
            # Check every 0.1 seconds
            await asyncio.sleep(0.1)

    # MAIN ORCHESTRATOR
    async def run(self):
        # --- 1. THE OUTER LOOP ---
        # This acts as the "Program Life Cycle".
        # Even if we disconnect, this loop forces us to start over.
        while True:
            try:
                print("\n--- Starting Session ---")

                # Clean up: Empty the audio queue so old questions don't play
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

                # --- 2. TAKE THE PHOTO ---
                # We do this here so a new photo is taken every time we restart.
                await asyncio.to_thread(snap)

                # --- 3. CONNECT TO GEMINI ---
                async with (
                    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session

                    # Start background tasks
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    # Start the button watcher (This is what triggers the restart)
                    tg.create_task(self.monitor_button())

                    print("Loading flashcard...")

                    # Read the image we just snapped
                    image_data = await asyncio.to_thread(
                        get_image_bytes, "/home/pi/flashcards/temp/card.jpg"
                    )

                    # Send Image + Prompt
                    await session.send_client_content(
                        turns=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        inline_data=types.Blob(
                                            data=image_data, mime_type="image/jpeg"
                                        )
                                    ),
                                    types.Part(
                                        text="Ask your first question about this flashcard."
                                    ),
                                ],
                            )
                        ]
                    )

                    # Keep the session alive until an error or button press
                    while True:
                        await asyncio.sleep(1)

            # --- 4. HANDLE THE RESTART ---
            except RestartSession:
                print("\n🔄 Resetting context and taking new photo...")
                # 'pass' implies: Do nothing, just exit the 'try' block.
                # Because we are inside 'while True', the code goes back to the top!
                pass

            except asyncio.CancelledError:
                pass

            except ExceptionGroup as EG:
                if self.audio_stream:
                    self.audio_stream.close()
                traceback.print_exception(EG)
                # If a real error happens (not a restart), we usually want to wait a bit
                await asyncio.sleep(1)


if __name__ == "__main__":
    # Initialize PyAudio globally with error suppression
    pya = pyaudio.PyAudio()

    try:
        main = AudioLoop()
        main.button.wait_for_press()

        snap()
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Cleanup audio drivers on exit
        pya.terminate()
