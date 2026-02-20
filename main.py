import asyncio
import traceback
import os
import io
import numpy as np
import sounddevice as sd
from scipy import signal
from PIL import Image
from gpiozero import Button, LED

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- HARDWARE CONFIG ---
BUTTON_PIN = 23
LED_PIN = 25
led = LED(LED_PIN)

# --- AUDIO CONFIGURATION ---
CHANNELS = 1
DTYPE = 'int16'
CHUNK_SIZE = 4096
HW_RATE = 48000         # AIY Voice HAT native rate
GEMINI_IN_RATE = 16000  # Gemini expects mic audio at this rate
GEMINI_OUT_RATE = 24000 # Gemini outputs audio at this rate

MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
client = genai.Client(
    http_options={"api_version": "v1alpha"},
    api_key=os.getenv("GEMINI_API_KEY")
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
    os.makedirs("/home/pi/flashcards/temp", exist_ok=True)
    os.system(
        "rpicam-still --width 1536 --height 1024 -o /home/pi/flashcards/temp/card.jpg -n --zsl"
    )
    print("Photo taken!")
    led.off()

def get_image_bytes(file_path):
    img = Image.open(file_path)
    img.thumbnail((2048, 2048))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()

class RestartSession(Exception):
    pass

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.button = Button(BUTTON_PIN)
        self.ai_is_speaking = False

        self.device_index = self.get_aiy_device()
        print(f"Hardware Ready: Device {self.device_index} at {HW_RATE}Hz")

    def get_aiy_device(self):
        """Finds the AIY Voice HAT or falls back to system default."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if 'aiy' in name or 'voice' in name or 'googlevoicehat' in name:
                return i
        return None  # fallback to system default

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if self.session is not None:
                await self.session.send_realtime_input(
                    media=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )

    async def listen_audio(self):
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time, status):
            if status:
                print(f"Mic Status: {status}")
            if not self.ai_is_speaking:
                # Downsample 48kHz → 16kHz (divide by 3)
                audio_48k = np.frombuffer(indata.tobytes(), dtype=DTYPE).astype(np.float32)
                audio_16k = signal.resample_poly(audio_48k, 1, 3).astype(DTYPE)
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": audio_16k.tobytes(), "mime_type": "audio/pcm;rate=16000"}
                )

        with sd.InputStream(device=self.device_index,
                            samplerate=HW_RATE,
                            channels=CHANNELS,
                            dtype=DTYPE,
                            callback=callback,
                            blocksize=CHUNK_SIZE):
            while True:
                await asyncio.sleep(1)

    async def receive_audio(self):
        while True:
            if self.session is None:
                await asyncio.sleep(0.1)
                continue

            async for response in self.session.receive():
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                if text := response.text:
                    print(text, end="", flush=True)

    async def play_audio(self):
        with sd.OutputStream(device=self.device_index,
                             samplerate=HW_RATE,
                             channels=CHANNELS,
                             dtype=DTYPE) as stream:
            while True:
                bytestream = await self.audio_in_queue.get()
                self.ai_is_speaking = True

                # Upsample 24kHz → 48kHz (multiply by 2)
                audio_24k = np.frombuffer(bytestream, dtype=DTYPE).astype(np.float32)
                audio_48k = signal.resample_poly(audio_24k, 2, 1).astype(DTYPE)
                audio_48k = (audio_48k * 0.1x).astype(DTYPE)  # 10% volume
                await asyncio.to_thread(stream.write, audio_48k)

                if self.audio_in_queue.empty():
                    await asyncio.sleep(0.2)
                    self.ai_is_speaking = False

    async def monitor_button(self):
        print(f"Button monitoring active on GPIO {BUTTON_PIN}...")
        while True:
            if self.button.is_pressed:
                print("\n!!! Button Pressed: Restarting Session !!!")
                await asyncio.sleep(1)
                raise RestartSession()
            await asyncio.sleep(0.1)

    async def run(self):
        while True:
            try:
                print("\n--- Starting Session ---")
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

                await asyncio.to_thread(snap)

                async with (
                    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())
                    tg.create_task(self.monitor_button())

                    print("Loading flashcard...")
                    image_data = await asyncio.to_thread(
                        get_image_bytes, "/home/pi/flashcards/temp/card.jpg"
                    )

                    await session.send_client_content(
                        turns=[types.Content(role="user", parts=[
                            types.Part(inline_data=types.Blob(data=image_data, mime_type="image/jpeg")),
                            types.Part(text="Ask your first question about this flashcard.")
                        ])]
                    )

                    while True:
                        await asyncio.sleep(1)

            except RestartSession:
                continue
            except Exception as e:
                print(f"\nSession Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(2)

if __name__ == "__main__":
    main = AudioLoop()
    print("Hold the AIY Button and release to take a photo!")
    main.button.wait_for_press()
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")
