from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep

from alpha_mini_rug.speech_to_text import SpeechToText

@inlineCallbacks
def text_to_speech(session, details, speech):
    # Let the robot say something:
    SpeechToText.do_speech = False
    yield session.call("rie.dialogue.say", text=speech)
