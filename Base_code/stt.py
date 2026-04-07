from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks, returnValue
from autobahn.twisted.util import sleep


@inlineCallbacks
def STT_continuous(session, audio_processor):
    while True:
        if not audio_processor.new_words:
            yield sleep(0.5)  # VERY IMPORTANT, OTHERWISE THE CONNECTION TO THE SERVER MIGHT CRASH
            print("waiting")
        else:
            word_array = audio_processor.give_me_words()
            print("I'm processing the words")
            print(word_array[-3:])  # print last 3 sentences
            returnValue(word_array[-1][0])
        audio_processor.loop()