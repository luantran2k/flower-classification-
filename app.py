from tkinter import *
from tkinter import filedialog as fd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageTk

window = Tk()
window.title("Test 2")
window.geometry("800x600")
window.configure(bg="#D0F4FF")
classifier = load_model('./flowermodel10epoch224.h5')
photo = './assets/img/loadImg.png'
window.photo = ImageTk.PhotoImage(Image.open(photo))
isPick = FALSE

def change_pic():
    import_filename = fd.askopenfilename()
    window.photo1 = ImageTk.PhotoImage(Image.open(import_filename))
    imgLabel.configure(image=window.photo1)
    img = Image.open(import_filename)
    img.save("./assets/img/imgTest.jpg")

imgLabel= Label(window,image=window.photo,bg="#D0F4FF")
imgLabel.config(width=600, height=400)
imgLabel.pack()

btnLoad = Button(window, text="Chọn ảnh", font=("Arial", 16), bg="#45BCFF", fg="white", padx="5px" , pady="5px", bd="0", command=change_pic)
btnLoad.place(relx=0.4, rely=0.9, anchor=CENTER)

txt = Label(window, text="", fg="#45BCFF", font=("Arial", 24), bg="#D0F4FF")
txt.place(relx=0.5, rely=0.8, anchor=CENTER)

def checkImage():
    test_image = image.load_img('assets/img/imgTest.jpg', target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    # print(np.argmax(result))
    # print(result)
    if result[0][0] == 1:
        prediction = 'hoa cúc'
    elif result[0][1] == 1:
        prediction = 'hoa bồ công anh'
    elif result[0][2] == 1:
        prediction = 'hoa hồng'
    elif result[0][3] == 1:
        prediction = 'hoa hướng dương'
    else:
        prediction = 'hoa tulip'
    txt.configure(text="Đây là " + prediction)

btnCheck = Button(window, text="Kiểm tra ảnh", font=("Arial", 16), bg="#45BCFF", fg="white", padx="5px" , pady="5px", bd="0", command=checkImage)
btnCheck.place(relx=0.6, rely=0.9, anchor=CENTER)

window.mainloop()
