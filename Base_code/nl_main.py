from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from google import genai
import os
from tts import text_to_speech
from stt import STT_continuous
from alpha_mini_rug.speech_to_text import SpeechToText
import cv2
import random
import evaluation_nl as eval



audio_processor = SpeechToText()

# Audio processing parameters
audio_processor.silence_time = 0.5
audio_processor.silence_threshold2 = 100
audio_processor.logging = True

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

def prompt(patient_level):
    prompt = f"""Je bent Mini Alpha, een vriendelijke sociale robot die helpt bij spraaktherapie voor mensen met expressieve afasie.
                Je spreekt langzaam, gebruikt makkelijke woorden en houdt je zinnen kort.
                Wees lief, geduldig en bemoedigend.

                Begin altijd met jezelf voor te stellen en de patiënt te vragen naar zijn naam. Stel daarna een klein gesprekje voor. 
                Als de patiënt alleen ja of nee kan zeggen ({patient_level} is "yes_no"), stel je alleen ja/nee-vragen zoals “Was je dag goed?” of “Hou je van muziek?”.
                Als de patiënt zinnen kan spreken ({patient_level} is "sentences"), kun je meer open vragen stellen zoals “Hoe was je dag?” of “Wat heb je voor lunch gegeten?”.
                Geef de patiënt genoeg tijd om te antwoorden voordat je weer spreekt.
                Jij moet niet de instructies voorlesen. Start met vragen naar de naam en wart voor een response.

                Als de patiënt gefrustreerd lijkt, lang doet over antwoorden, of moeite heeft met een woord, erken hun inspanning en bemoedig ze.
                Zeg dingen zoals “Ik zie dat je gefrustreerd bent, maar je doet het geweldig” of “Ik zie dat je hard je best doet, ga zo door.”
                Geef niet te snel hints.
                Als hulp nodig is, geef eerst een kort en eenvoudig voorbeeldzinnetje met een lege plek voor het ontbrekende woord (bijvoorbeeld: “Ik zie een ...”).
                Als dat niet werkt, geef de eerste letter van het woord.
                Voeg daarna de volgende letter toe als nodig, totdat ze het woord zelf kunnen zeggen.
                Hun spraak wordt vaak beter naarmate ze blijven praten, dus wees geduldig en ondersteunend.

                Na een kort gesprekje zeg je dat jullie samen een oefening gaan doen.
                Leg het duidelijk uit in korte zinnen.
                Zeg dat je een foto op het scherm laat zien en dat hun taak is te beschrijven wat ze zien.
                Controleer of ze begrijpen wat ze moeten doen, met ja/nee-vragen voor ja/nee-patiënten of open vragen voor zin-patiënten.
                Zorg ervoor dat je zegt: "Laten we beginnen met de oefening!", als je de oefening start. En dan iets als: "hier is de foto"

                Als de patiënt spreekt, luister en vergelijk hun antwoord met een richtlijn (uit een andere prompt).
                Geef lieve en eenvoudige feedback zoals “Goed gedaan”, “Dat is bijna goed!”, of “Ik zie dat je heel hard je best deed.”
                Als de patiënt de foto goed beschrijft, of als je merkt dat ze niet weten hoe verder te beschrijven, ga door naar de tweede foto. Je moet de zin "volgende foto" gebruiken daarvoor.

                Als de oefening klaar is, of als de patiënt wil stoppen, neem afscheid.
                Zeg altijd “Dag” aan het eind.
                Bijvoorbeeld:
                “Goed gedaan vandaag! Je hebt hard gewerkt. Fijn om met je te praten. Dag!”



                Variabelen
                {patient_level}: "yes_no" of "sentences"
                """
    return prompt

seen_images = set()

def pick_img_and_display(image_files):
    # picks and image and displays it
    available = [img_path for img_path in image_files if img_path not in seen_images]
    if not available:
        print("No new images available.")
        return None
    
    img_path = random.choice(available)
    img_filename = os.path.basename(img_path)
    img_name = os.path.splitext(img_filename)[0]
    
    img = cv2.imread(img_path)
    cv2.imshow(img_name, img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    seen_images.add(img_path)  # Track this image
    print(f"Selected image: {img_name}")
    return img_name

@inlineCallbacks
def main(session, details):
      
        yield session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield session.call("rom.sensor.hearing.sensitivity", 2000)
        yield session.call("rie.dialogue.config.language", lang="nl")
        yield session.subscribe(audio_processor.listen_continues, "rom.sensor.hearing.stream")
        yield session.call("rom.sensor.hearing.stream")

        exercise = False
        patient_level = "yes_no"  # or "sentences"
        system_instruction = prompt(patient_level)
        # Create a chat session
        chat = client.chats.create(model="gemini-3-flash-preview")
        print("Mini Alpha is ready. Type 'exit' to stop.\n")

        # First message from Mini Alpha
        first = chat.send_message(system_instruction)
        yield text_to_speech(session, details, first.text)
        sleep(0.5)
        

        while True:
            SpeechToText.do_speech_recognition = True
            user_msg = yield STT_continuous(session, audio_processor)
            print(user_msg)
            # goodybe check
            if user_msg.lower().strip() in ("exit", "quit", "goodbye"):
                goodbye = chat.send_message("The patient wants to stop. Say goodbye.")
                yield text_to_speech(session, details, goodbye.text)
                break

            response = chat.send_message(user_msg)
            print(response.text)
            yield text_to_speech(session, details, response.text)
            # stop check
            if response.text.lower().strip() in ("stop", "einde", "dag"):
                break
            
            # start/ continue exercise and show image
            if ("laten we beginnen met de oefening" in response.text.lower() or "volgende foto" in response.text.lower()):
                exercise = True
                image_dir = "images"
                image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                img_name = pick_img_and_display(image_files)
                if img_name is None:
                    # if no more images, end the gameplay
                    goodbye = chat.send_message("Zeg tegen de patiënt: 'We hebben genoeg geoefend vandaag.'")
                    yield text_to_speech(session, details, goodbye.text)
                    break
                else:
                    # otherwise show image
                    eval_func = getattr(eval, img_name, None)
                    if eval_func:
                        eval_prompt = eval_func()
                        chat.send_message(eval_prompt)
                    else:
                        raise ValueError



        session.leave() # Close the connection with the robot



wamp = Component(
	transports=[{
		"url": "ws://wamp.robotsindeklas.nl",
		"serializers": ["msgpack"],
		"max_retries": 0
	}],
	realm="rie.69c14943a31f42c33c3f9ebc",
)

wamp.on_join(main)

if __name__ == "__main__":
	run([wamp])