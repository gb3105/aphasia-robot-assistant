from hint_model.features import FeatureExtractor
import time

extractor = FeatureExtractor(model_size="base", language="en")
extractor.start()

print("Speak into your mic for 10 seconds...")
for i in range(10):
    time.sleep(1)
    result = extractor.get_latest_features()
    if result[0] is not None:
        features, transcript = result
        print(f"[{i+1}s] Transcript: '{transcript}'")
        print(f"      Silence: {features[0]:.2f}s | Pauses: {features[1]:.0f} | Rate: {features[2]:.2f} w/s")

extractor.stop()