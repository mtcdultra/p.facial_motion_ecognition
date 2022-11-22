from deepface import DeepFace

obj = DeepFace.analyze(img_path="ft2.jpg", actions=["age","gender","emotion"])

print(obj)