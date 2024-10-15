import tensorflow as tf

from src.predict_hairstyle import load_image_to_predict, preprocess_image_for_model, make_prediction


model_path = r"C:\Users\user\Projekty\BaldOrNot\trainings\training_name2024-10-14_23-34-02\model.keras"

image_dir_path = r"C:\Users\user\Projekty\BaldOrNot\scrapping\downloaded_images"
image_name = "image_89.jpg"

image = load_image_to_predict(image_dir_path, image_name)
preprocessed_image = preprocess_image_for_model(image)

model = tf.keras.models.load_model(model_path)

prediction = make_prediction(model, preprocessed_image)
print(f'Predicted label: {prediction}')



