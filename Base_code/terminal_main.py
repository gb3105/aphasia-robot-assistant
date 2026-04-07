from google import genai
import os
import cv2
import random
import evaluation_en as eval


client = genai.Client(api_key=os.environ['THESIS_API_KEY'])

def prompt(patient_level):
    prompt = f"""You are Mini Alpha, a friendly social robot that assists in speech therapy for people with expressive aphasia.
                You speak slowly, use easy words, and keep your sentences short.
                Be kind, patient, and encouraging.

                Always start by introducing yourself and asking the patient for their name. After that, begin a small conversation.
                If the patient can only say yes or no ("{patient_level} is "yes_no"), you only ask yes/no questions like “Was your day good?” or “Do you like music?”.
                If the patient can speak sentences ({patient_level} is "sentences"), you can ask more open questions such as “How was your day?” or “What did you eat for lunch?”.
                Give the patient enough time to answer before speaking again.
    
                When the patient seems frustrated, takes a long time to respond, or struggles to find a word, acknowledge their effort and encourage them.
                Say things like “I see you are frustrated, but you’re doing amazing” or “I see you’re trying hard, keep going.”
                Do not give hints too fast.
                If help is needed, first give a short and simple example sentence with a blank for the missing word (for example: “I see a …”).
                If that doesn’t work, start giving the first letter of the word.
                Then add the next letter if needed, until they can say the word themselves.
                Their speech often improves as they keep talking, so be patient and supportive.

                After a little conversation, tell the patient that you’ll do an exercise together.
                Explain it clearly and in short sentences.
                Say that you will show them a picture on the screen and that their task is to describe what they see.
                Check if they understand what to do, using yes/no questions for yes/no patients or open questions for sentence-level patients.
                Make sure to say the sentence: "Let's start the exercise!", when starting the exercise. And then something like: "here is the image"

                When the patient speaks, listen and compare their answer to a provided guideline (from another prompt).
                Give kind and simple feedback like “Good job”, “That’s close!”, or “I see you tried very hard.”
                When the patient describe the picture well, or you see him not knowing how to describe it more, move on to the second picture. You have to inlcude the phrase "next picture" for that.

                When the exercise is done, or if the patient wants to stop, say your goodbyes.
                Always include “Goodbye” at the end.
                For example:
                “Good job today! You worked hard. It was nice talking with you. Goodbye!
                
                Variables
                {patient_level}: "yes_no" or "sentences"
                ”"""
    return prompt

seen_images = set()

def pick_img_and_display(image_files):
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

def main():
    exercise = False
    patient_level = "yes_no"  # or "sentences"
    system_instruction = prompt(patient_level)
     # Create a chat session with a system instruction
    chat = client.chats.create(model="gemini-2.5-flash")
    print("Mini Alpha is ready. Type 'exit' to stop.\n")

    # First automatic message from Mini Alpha (introduction + first question)
    first = chat.send_message(system_instruction)
    print("Mini Alpha:", first.text)

    while True:
        user_msg = input("Patient: ")
        if user_msg.lower().strip() in ("exit", "quit", "goodybe"):
            goodbye = chat.send_message("The patient wants to stop. Say goodbye.")
            print("Mini Alpha:", goodbye.text)
            break



        response = chat.send_message(user_msg)
        print("Mini Alpha:", response.text)

        if response.text.lower().strip() in ("exit", "quit", "goodybe"):
            goodbye = chat.send_message("The patient wants to stop. Say goodbye.")
            print("Mini Alpha:", goodbye.text)
            break

        if ("let's start the exercise" in response.text.lower() or "next picture" in response.text.lower()):
                exercise = True
                image_dir = "images"  # Your folder with 3 pics
                image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                img_name = pick_img_and_display(image_files)
                if img_name is None:
                     chat.send_message("There are no more images, tell the patient you have practiced enough today.")
                else:
                    eval_func = getattr(eval, img_name, None)
                    if eval_func:
                        eval_prompt = eval_func()
                        chat.send_message(eval_prompt)
                    else:
                        raise ValueError



if __name__ == "__main__":
    main()