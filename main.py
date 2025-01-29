import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, Frame
from PIL import Image, ImageTk
from groq import Groq

# Wczytanie modelu
model_path = "plant_classifier.h5"
model = tf.keras.models.load_model(model_path)

# Wczytanie nazw roślin z pliku JSON
json_path = "plantnet300K_species_id_2_name.json"
with open(json_path, "r", encoding="utf-8") as file:
    plant_names = json.load(file)

# Tworzenie listy kluczy do poprawnego mapowania indeksów klas na identyfikatory roślin
plant_keys = list(plant_names.keys())

def give_short_plant_info(plant_name):
    # Klucz API może zostać usunięty, aby pobrać klucz należy wejść na stronę Groq i wygenerować nowy klucz
    client = Groq(
        api_key="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",  # Pobieranie klucza API
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f"Podaj informacje na temat gatunku rośliny {plant_name} oraz wskazówki dotyczące pielęgnacji."}
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content if chat_completion.choices else "Brak informacji."


def classify_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Dopasowanie do modelu
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizacja

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)  # Indeks przewidywanej klasy

    if class_idx < len(plant_keys):  # Upewnienie się, że indeks mieści się w zakresie
        plant_id = plant_keys[class_idx]
        class_name = plant_names.get(plant_id, "Nieznana roślina")
    else:
        class_name = "Nieznana roślina"

    return class_name


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(150, 150, image=img_tk, anchor=tk.CENTER)
        canvas.image = img_tk

        class_name = classify_image(file_path)
        care_info = give_short_plant_info(class_name)
        label_result.config(text=f"Roślina: {class_name}")
        label_care.config(text=f"Pielęgnacja: {care_info}")


def exit_app():
    root.destroy()


# Tworzenie GUI
root = tk.Tk()
root.title("Rozpoznawanie roślin")
root.geometry("800x600")

main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Lewa strona - wyświetlanie zdjęcia
left_frame = Frame(main_frame, width=400, height=400)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas = Canvas(left_frame, width=300, height=300)
canvas.pack(pady=20)

# Prawa strona - informacje o roślinie
right_frame = Frame(main_frame, width=400, height=400)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
label_result = Label(right_frame, text="Roślina: ", font=("Arial", 14))
label_result.pack(pady=10)
label_care = Label(right_frame, text="Pielęgnacja: ", font=("Arial", 8), wraplength=350)
label_care.pack()

# Dolna sekcja z przyciskami
bottom_frame = Frame(root)
bottom_frame.pack(side=tk.BOTTOM, pady=10)
btn_upload = Button(bottom_frame, text="Wybierz zdjęcie", command=upload_image)
btn_upload.pack(side=tk.LEFT, padx=10)
btn_exit = Button(bottom_frame, text="Koniec", command=exit_app)
btn_exit.pack(side=tk.RIGHT, padx=10)

root.mainloop()
