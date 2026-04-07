How to prevent robot from hearing itself when it speaks

extractor.robot_speaking_start()
yield text_to_speech(session, details, response.text)
extractor.robot_speaking_end()